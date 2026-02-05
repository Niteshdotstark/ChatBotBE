"""
Enhanced Semantic Search System for RAG Chatbot
Implements configurable similarity thresholds, result ranking, diversity scoring, and confidence calculation.
"""

import math
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from langchain_core.documents import Document

from rag_model.config_models import RetrievalConfig


class ConfidenceLevel(Enum):
    """Confidence levels for search results"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class SearchResult:
    """Enhanced search result with confidence and ranking information"""
    document: Document
    similarity_score: float
    relevance_score: float
    recency_score: float
    diversity_score: float
    confidence_level: ConfidenceLevel
    rank: int
    uncertainty_indicators: List[str]


@dataclass
class SearchMetrics:
    """Metrics for search quality assessment"""
    total_results: int
    filtered_results: int
    average_similarity: float
    average_confidence: float
    diversity_index: float
    threshold_used: float
    adaptive_adjustments: List[str]


class SimilarityThresholdManager:
    """
    Manages configurable similarity thresholds with adaptive adjustment.
    Implements Requirements 3.1 for threshold-based filtering and adaptive adjustment.
    """
    
    def __init__(self, base_threshold: float = 0.7):
        self.base_threshold = base_threshold
        self.min_threshold = 0.5
        self.max_threshold = 0.95
        self.adjustment_history = []
    
    def get_adaptive_threshold(
        self, 
        initial_results: List[Document], 
        target_result_count: int = 5,
        query_complexity: str = "moderate"
    ) -> Tuple[float, List[str]]:
        """
        Calculate adaptive similarity threshold based on initial results and query characteristics.
        Implements Requirements 3.1 for adaptive threshold adjustment.
        """
        
        adjustments = []
        threshold = self.base_threshold
        
        if not initial_results:
            # No results - lower threshold significantly
            threshold = max(self.min_threshold, self.base_threshold - 0.2)
            adjustments.append("lowered_threshold_no_results")
            return threshold, adjustments
        
        # Calculate similarity score statistics
        similarity_scores = []
        for doc in initial_results:
            if hasattr(doc, 'metadata') and 'similarity_score' in doc.metadata:
                # Convert distance to similarity (assuming cosine distance)
                distance = doc.metadata['similarity_score']
                similarity = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
                similarity_scores.append(similarity)
        
        if not similarity_scores:
            # No similarity scores available - use base threshold
            return threshold, adjustments
        
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        max_similarity = max(similarity_scores)
        min_similarity = min(similarity_scores)
        
        # Adjust based on result quality
        if len(initial_results) < target_result_count:
            if avg_similarity < self.base_threshold:
                # Lower threshold to get more results
                threshold = max(self.min_threshold, min(avg_similarity - 0.05, self.base_threshold - 0.1))
                adjustments.append("lowered_threshold_insufficient_results")
        
        elif len(initial_results) > target_result_count * 2:
            if avg_similarity > self.base_threshold:
                # Raise threshold to get more selective results
                threshold = min(self.max_threshold, max(avg_similarity + 0.05, self.base_threshold + 0.1))
                adjustments.append("raised_threshold_too_many_results")
        
        # Adjust based on query complexity
        complexity_adjustments = {
            "simple": 0.05,      # Simple queries can be more selective
            "moderate": 0.0,     # No adjustment
            "complex": -0.05,    # Complex queries need broader search
            "analytical": -0.1   # Analytical queries need comprehensive results
        }
        
        complexity_adj = complexity_adjustments.get(query_complexity, 0.0)
        if complexity_adj != 0.0:
            threshold = max(self.min_threshold, min(self.max_threshold, threshold + complexity_adj))
            adjustments.append(f"adjusted_for_{query_complexity}_complexity")
        
        # Adjust based on score distribution
        score_range = max_similarity - min_similarity
        if score_range < 0.1:
            # Very similar scores - might need to be more selective
            threshold = min(self.max_threshold, threshold + 0.05)
            adjustments.append("raised_threshold_similar_scores")
        elif score_range > 0.4:
            # Wide score range - can be more permissive
            threshold = max(self.min_threshold, threshold - 0.05)
            adjustments.append("lowered_threshold_diverse_scores")
        
        # Ensure threshold is within bounds
        threshold = max(self.min_threshold, min(self.max_threshold, threshold))
        
        # Record adjustment
        self.adjustment_history.append({
            "original_threshold": self.base_threshold,
            "adjusted_threshold": threshold,
            "adjustments": adjustments,
            "result_count": len(initial_results),
            "avg_similarity": avg_similarity
        })
        
        return threshold, adjustments
    
    def filter_by_threshold(
        self, 
        documents: List[Document], 
        threshold: float
    ) -> Tuple[List[Document], int]:
        """
        Filter documents by similarity threshold.
        Implements Requirements 3.1 for threshold-based filtering.
        """
        
        filtered_docs = []
        filtered_count = 0
        
        for doc in documents:
            if hasattr(doc, 'metadata') and 'similarity_score' in doc.metadata:
                # Convert distance to similarity
                distance = doc.metadata['similarity_score']
                similarity = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
                
                if similarity >= threshold:
                    # Update metadata with similarity score
                    doc.metadata['similarity_score'] = similarity
                    filtered_docs.append(doc)
                else:
                    filtered_count += 1
            else:
                # No similarity score - include by default
                filtered_docs.append(doc)
        
        return filtered_docs, filtered_count


class ResultRankingSystem:
    """
    Implements result ranking based on relevance, recency, and diversity.
    Implements Requirements 3.2, 3.4 for ranking and diversity scoring.
    """
    
    def __init__(self):
        self.relevance_weight = 0.6
        self.recency_weight = 0.2
        self.diversity_weight = 0.2
    
    def rank_results(
        self, 
        documents: List[Document], 
        query: str,
        diversity_threshold: float = 0.8
    ) -> List[SearchResult]:
        """
        Rank search results by relevance, recency, and diversity.
        Implements Requirements 3.2, 3.4 for relevance/recency ranking and diversity scoring.
        """
        
        if not documents:
            return []
        
        # Calculate individual scores
        relevance_scores = self._calculate_relevance_scores(documents, query)
        recency_scores = self._calculate_recency_scores(documents)
        diversity_scores = self._calculate_diversity_scores(documents, diversity_threshold)
        
        # Create SearchResult objects
        search_results = []
        for i, doc in enumerate(documents):
            relevance = relevance_scores[i]
            recency = recency_scores[i]
            diversity = diversity_scores[i]
            
            # Calculate composite score
            composite_score = (
                self.relevance_weight * relevance +
                self.recency_weight * recency +
                self.diversity_weight * diversity
            )
            
            # Determine confidence level
            similarity = doc.metadata.get('similarity_score', 0.0)
            confidence_level = self._determine_confidence_level(similarity, composite_score)
            
            # Identify uncertainty indicators
            uncertainty_indicators = self._identify_uncertainty_indicators(doc, similarity, composite_score)
            
            search_result = SearchResult(
                document=doc,
                similarity_score=similarity,
                relevance_score=relevance,
                recency_score=recency,
                diversity_score=diversity,
                confidence_level=confidence_level,
                rank=i + 1,  # Will be updated after sorting
                uncertainty_indicators=uncertainty_indicators
            )
            
            search_results.append(search_result)
        
        # Sort by composite score (descending)
        search_results.sort(key=lambda x: (
            self.relevance_weight * x.relevance_score +
            self.recency_weight * x.recency_score +
            self.diversity_weight * x.diversity_score
        ), reverse=True)
        
        # Update ranks
        for i, result in enumerate(search_results):
            result.rank = i + 1
        
        return search_results
    
    def _calculate_relevance_scores(self, documents: List[Document], query: str) -> List[float]:
        """Calculate relevance scores based on content similarity and keyword matching"""
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        relevance_scores = []
        
        for doc in documents:
            # Base relevance from similarity score
            base_relevance = doc.metadata.get('similarity_score', 0.5)
            
            # Keyword matching bonus
            content_lower = doc.page_content.lower()
            content_words = set(re.findall(r'\b\w+\b', content_lower))
            
            # Calculate keyword overlap
            common_words = query_words.intersection(content_words)
            keyword_overlap = len(common_words) / len(query_words) if query_words else 0
            
            # Exact phrase matching bonus
            phrase_bonus = 0.0
            if len(query.split()) > 1:
                if query_lower in content_lower:
                    phrase_bonus = 0.2
            
            # Content length normalization (prefer comprehensive but not overly long content)
            content_length = len(doc.page_content)
            length_factor = 1.0
            if content_length < 100:
                length_factor = 0.8  # Penalize very short content
            elif content_length > 2000:
                length_factor = 0.9  # Slightly penalize very long content
            
            # Combine factors
            relevance = (
                base_relevance * 0.7 +
                keyword_overlap * 0.2 +
                phrase_bonus +
                (length_factor - 1.0) * 0.1
            )
            
            relevance_scores.append(max(0.0, min(1.0, relevance)))
        
        return relevance_scores
    
    def _calculate_recency_scores(self, documents: List[Document]) -> List[float]:
        """Calculate recency scores based on document metadata"""
        
        recency_scores = []
        
        for doc in documents:
            # Default recency score
            recency = 0.5
            
            # Check for timestamp information in metadata
            metadata = doc.metadata
            
            # Look for various timestamp fields
            timestamp_fields = ['timestamp', 'created_at', 'modified_at', 'date', 'last_updated']
            
            for field in timestamp_fields:
                if field in metadata:
                    try:
                        # Simple heuristic: more recent = higher score
                        # This is a placeholder - in practice, you'd parse actual timestamps
                        timestamp_str = str(metadata[field])
                        if '2024' in timestamp_str or '2025' in timestamp_str:
                            recency = 0.9
                        elif '2023' in timestamp_str:
                            recency = 0.7
                        elif '2022' in timestamp_str:
                            recency = 0.5
                        else:
                            recency = 0.3
                        break
                    except:
                        continue
            
            # Check source type for recency hints
            source = metadata.get('source', '')
            if 'recent' in source.lower() or 'latest' in source.lower():
                recency = min(1.0, recency + 0.2)
            
            recency_scores.append(recency)
        
        return recency_scores
    
    def _calculate_diversity_scores(self, documents: List[Document], diversity_threshold: float) -> List[float]:
        """
        Calculate diversity scores to ensure result variety.
        Implements Requirements 3.4 for diversity scoring for ambiguous queries.
        """
        
        diversity_scores = []
        
        if len(documents) <= 1:
            return [1.0] * len(documents)
        
        # Extract key features for diversity calculation
        document_features = []
        for doc in documents:
            features = {
                'source': doc.metadata.get('source', ''),
                'content_length': len(doc.page_content),
                'key_terms': set(re.findall(r'\b\w{4,}\b', doc.page_content.lower())[:10])
            }
            document_features.append(features)
        
        # Calculate diversity for each document
        for i, doc_features in enumerate(document_features):
            diversity_score = 1.0
            
            # Compare with other documents
            for j, other_features in enumerate(document_features):
                if i == j:
                    continue
                
                # Source diversity
                source_similarity = 1.0 if doc_features['source'] == other_features['source'] else 0.0
                
                # Content length similarity
                len_diff = abs(doc_features['content_length'] - other_features['content_length'])
                length_similarity = max(0.0, 1.0 - len_diff / 1000.0)
                
                # Term overlap similarity
                common_terms = doc_features['key_terms'].intersection(other_features['key_terms'])
                term_similarity = len(common_terms) / max(len(doc_features['key_terms']), 1)
                
                # Combined similarity
                combined_similarity = (source_similarity * 0.3 + length_similarity * 0.2 + term_similarity * 0.5)
                
                # Reduce diversity score based on similarity
                if combined_similarity > diversity_threshold:
                    diversity_score *= (1.0 - combined_similarity * 0.5)
            
            diversity_scores.append(max(0.1, diversity_score))
        
        return diversity_scores
    
    def _determine_confidence_level(self, similarity: float, composite_score: float) -> ConfidenceLevel:
        """Determine confidence level based on similarity and composite scores"""
        
        # Weighted confidence calculation
        confidence = similarity * 0.7 + composite_score * 0.3
        
        if confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _identify_uncertainty_indicators(
        self, 
        document: Document, 
        similarity: float, 
        composite_score: float
    ) -> List[str]:
        """Identify indicators of uncertainty in search results"""
        
        indicators = []
        
        # Low similarity score
        if similarity < 0.6:
            indicators.append("low_similarity")
        
        # Low composite score
        if composite_score < 0.5:
            indicators.append("low_overall_score")
        
        # Short content
        if len(document.page_content) < 100:
            indicators.append("limited_content")
        
        # Vague or generic content
        content_lower = document.page_content.lower()
        vague_indicators = ['maybe', 'possibly', 'might', 'could be', 'uncertain', 'unclear']
        if any(indicator in content_lower for indicator in vague_indicators):
            indicators.append("vague_content")
        
        # Missing key information
        if 'incomplete' in content_lower or 'partial' in content_lower:
            indicators.append("incomplete_information")
        
        return indicators


class ConfidenceCalculator:
    """
    Calculates search confidence and provides uncertainty indication.
    Implements Requirements 3.5 for confidence scoring and uncertainty indication.
    """
    
    def __init__(self):
        self.confidence_factors = {
            'similarity_weight': 0.4,
            'result_count_weight': 0.2,
            'diversity_weight': 0.2,
            'consistency_weight': 0.2
        }
    
    def calculate_search_confidence(
        self, 
        search_results: List[SearchResult],
        query: str,
        original_result_count: int
    ) -> Tuple[float, List[str]]:
        """
        Calculate overall search confidence and identify uncertainty indicators.
        Implements Requirements 3.5 for search confidence calculation and uncertainty indication.
        """
        
        if not search_results:
            return 0.0, ["no_results_found"]
        
        uncertainty_indicators = []
        
        # Factor 1: Average similarity confidence
        avg_similarity = sum(result.similarity_score for result in search_results) / len(search_results)
        similarity_confidence = min(1.0, avg_similarity * 1.2)  # Boost slightly
        
        # Factor 2: Result count confidence
        result_count_confidence = self._calculate_result_count_confidence(
            len(search_results), original_result_count
        )
        
        if len(search_results) < 3:
            uncertainty_indicators.append("few_results")
        
        # Factor 3: Diversity confidence
        diversity_confidence = self._calculate_diversity_confidence(search_results)
        
        if diversity_confidence < 0.5:
            uncertainty_indicators.append("low_diversity")
        
        # Factor 4: Consistency confidence
        consistency_confidence = self._calculate_consistency_confidence(search_results, query)
        
        if consistency_confidence < 0.6:
            uncertainty_indicators.append("inconsistent_results")
        
        # Combine factors
        overall_confidence = (
            self.confidence_factors['similarity_weight'] * similarity_confidence +
            self.confidence_factors['result_count_weight'] * result_count_confidence +
            self.confidence_factors['diversity_weight'] * diversity_confidence +
            self.confidence_factors['consistency_weight'] * consistency_confidence
        )
        
        # Additional uncertainty indicators
        high_uncertainty_count = sum(
            1 for result in search_results 
            if result.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]
        )
        
        if high_uncertainty_count > len(search_results) / 2:
            uncertainty_indicators.append("many_low_confidence_results")
            overall_confidence *= 0.8
        
        # Check for conflicting information
        if self._detect_conflicting_information(search_results):
            uncertainty_indicators.append("conflicting_information")
            overall_confidence *= 0.9
        
        return max(0.0, min(1.0, overall_confidence)), uncertainty_indicators
    
    def _calculate_result_count_confidence(self, filtered_count: int, original_count: int) -> float:
        """Calculate confidence based on result count"""
        
        if original_count == 0:
            return 0.0
        
        # Optimal range is 3-10 results
        if 3 <= filtered_count <= 10:
            return 1.0
        elif filtered_count < 3:
            return filtered_count / 3.0
        else:
            # Too many results might indicate low selectivity
            return max(0.5, 1.0 - (filtered_count - 10) / 20.0)
    
    def _calculate_diversity_confidence(self, search_results: List[SearchResult]) -> float:
        """Calculate confidence based on result diversity"""
        
        if len(search_results) <= 1:
            return 1.0
        
        avg_diversity = sum(result.diversity_score for result in search_results) / len(search_results)
        return avg_diversity
    
    def _calculate_consistency_confidence(self, search_results: List[SearchResult], query: str) -> float:
        """Calculate confidence based on result consistency"""
        
        if len(search_results) <= 1:
            return 1.0
        
        # Check similarity score variance
        similarities = [result.similarity_score for result in search_results]
        avg_similarity = sum(similarities) / len(similarities)
        variance = sum((s - avg_similarity) ** 2 for s in similarities) / len(similarities)
        
        # Lower variance = higher consistency
        consistency = max(0.0, 1.0 - variance * 2.0)
        
        return consistency
    
    def _detect_conflicting_information(self, search_results: List[SearchResult]) -> bool:
        """Detect if search results contain conflicting information"""
        
        # Simple heuristic: look for contradictory terms
        contradictory_pairs = [
            ('yes', 'no'), ('true', 'false'), ('correct', 'incorrect'),
            ('valid', 'invalid'), ('supported', 'unsupported'),
            ('recommended', 'not recommended'), ('safe', 'unsafe')
        ]
        
        all_content = ' '.join(result.document.page_content.lower() for result in search_results)
        
        for term1, term2 in contradictory_pairs:
            if term1 in all_content and term2 in all_content:
                return True
        
        return False
    
    def generate_uncertainty_message(self, uncertainty_indicators: List[str], confidence: float) -> str:
        """Generate user-friendly uncertainty message"""
        
        if confidence >= 0.8:
            return ""
        
        messages = []
        
        if "no_results_found" in uncertainty_indicators:
            messages.append("I couldn't find relevant information about that.")
        elif "few_results" in uncertainty_indicators:
            messages.append("I found limited information about this topic.")
        elif "low_similarity" in uncertainty_indicators:
            messages.append("The available information might not be directly related to your question.")
        elif "inconsistent_results" in uncertainty_indicators:
            messages.append("I found some conflicting information on this topic.")
        elif "many_low_confidence_results" in uncertainty_indicators:
            messages.append("I'm not very confident about the available information.")
        
        if confidence < 0.5:
            messages.append("You might want to rephrase your question or ask for more specific information.")
        
        return " ".join(messages)


class SearchScopeExpander:
    """
    Handles search scope expansion and alternative query suggestions.
    Implements Requirements 3.3 for expanding search scope when results are insufficient.
    """
    
    def __init__(self):
        self.expansion_strategies = [
            "lower_threshold",
            "increase_top_k", 
            "broaden_query",
            "alternative_terms"
        ]
    
    def should_expand_search(
        self, 
        search_results: List[SearchResult], 
        original_query: str,
        min_results: int = 3
    ) -> Tuple[bool, List[str]]:
        """
        Determine if search scope should be expanded and suggest strategies.
        Implements Requirements 3.3 for detecting insufficient search results.
        """
        
        expansion_reasons = []
        should_expand = False
        
        # Check result count
        if len(search_results) < min_results:
            should_expand = True
            expansion_reasons.append("insufficient_result_count")
        
        # Check result quality
        if search_results:
            avg_confidence = sum(
                1.0 if result.confidence_level == ConfidenceLevel.HIGH else
                0.7 if result.confidence_level == ConfidenceLevel.MEDIUM else
                0.4 if result.confidence_level == ConfidenceLevel.LOW else 0.2
                for result in search_results
            ) / len(search_results)
            
            if avg_confidence < 0.5:
                should_expand = True
                expansion_reasons.append("low_quality_results")
        
        # Check for very specific queries that might need broadening
        query_words = original_query.split()
        if len(query_words) > 8:  # Very specific query
            should_expand = True
            expansion_reasons.append("overly_specific_query")
        
        return should_expand, expansion_reasons
    
    def expand_search_scope(
        self, 
        original_query: str,
        tenant_id: int,
        current_config: RetrievalConfig,
        expansion_reasons: List[str],
        retrieval_function: Callable = None
    ) -> Tuple[List[Document], RetrievalConfig, List[str]]:
        """
        Expand search scope using various strategies.
        Implements Requirements 3.3 for search scope expansion.
        """
        
        if retrieval_function is None:
            try:
                from rag_model.rag_utils import retrieve_s3_vectors
                retrieval_function = retrieve_s3_vectors
            except ImportError:
                raise ValueError("No retrieval function provided and rag_utils not available")
        
        expanded_results = []
        strategies_used = []
        expanded_config = RetrievalConfig(
            base_top_k=current_config.base_top_k,
            max_top_k=current_config.max_top_k,
            min_similarity_threshold=current_config.min_similarity_threshold,
            diversity_threshold=current_config.diversity_threshold,
            rerank_enabled=current_config.rerank_enabled,
            hybrid_search_weight=current_config.hybrid_search_weight
        )
        
        # Strategy 1: Lower similarity threshold
        if "low_quality_results" in expansion_reasons or "insufficient_result_count" in expansion_reasons:
            expanded_config.min_similarity_threshold = max(0.5, current_config.min_similarity_threshold - 0.15)
            strategies_used.append("lowered_threshold")
        
        # Strategy 2: Increase top_k
        if "insufficient_result_count" in expansion_reasons:
            expanded_config.base_top_k = min(current_config.max_top_k, current_config.base_top_k + 5)
            strategies_used.append("increased_top_k")
        
        # Strategy 3: Broaden query terms
        if "overly_specific_query" in expansion_reasons:
            broader_query = self._broaden_query(original_query)
            expanded_results = retrieval_function(broader_query, tenant_id, expanded_config.base_top_k)
            strategies_used.append("broadened_query")
        else:
            expanded_results = retrieval_function(original_query, tenant_id, expanded_config.base_top_k)
        
        return expanded_results, expanded_config, strategies_used
    
    def _broaden_query(self, query: str) -> str:
        """Broaden query by removing very specific terms and adding general terms"""
        
        words = query.split()
        
        # Remove very specific terms (numbers, dates, specific names)
        general_words = []
        for word in words:
            # Keep general terms, remove very specific ones
            if not (word.isdigit() or 
                   re.match(r'\d{4}', word) or  # Years
                   len(word) > 15 or  # Very long specific terms
                   word.isupper() and len(word) > 3):  # Acronyms
                general_words.append(word)
        
        # Add general context terms
        if len(general_words) < len(words):
            general_words.extend(["information", "details", "overview"])
        
        return " ".join(general_words[:8])  # Limit to 8 words
    
    def suggest_alternative_queries(self, original_query: str) -> List[str]:
        """
        Suggest alternative query formulations.
        Implements Requirements 3.3 for alternative query suggestions.
        """
        
        alternatives = []
        query_lower = original_query.lower()
        
        # Suggest broader versions
        if "how to" in query_lower:
            alternatives.append(original_query.replace("how to", "guide for"))
            alternatives.append(original_query.replace("how to", "steps to"))
        
        if "what is" in query_lower:
            alternatives.append(original_query.replace("what is", "definition of"))
            alternatives.append(original_query.replace("what is", "explanation of"))
        
        # Suggest more specific versions
        words = original_query.split()
        if len(words) < 4:
            alternatives.append(f"{original_query} examples")
            alternatives.append(f"{original_query} tutorial")
            alternatives.append(f"{original_query} guide")
        
        # Suggest related terms
        if "error" in query_lower:
            alternatives.append(original_query.replace("error", "problem"))
            alternatives.append(original_query.replace("error", "issue"))
        
        if "configure" in query_lower:
            alternatives.append(original_query.replace("configure", "setup"))
            alternatives.append(original_query.replace("configure", "install"))
        
        return alternatives[:3]  # Return top 3 alternatives


class EnhancedSemanticSearch:
    """
    Main enhanced semantic search system that combines all components.
    Implements the complete enhanced semantic search from Requirements 3.1, 3.2, 3.3, 3.4, 3.5.
    """
    
    def __init__(self, config: RetrievalConfig = None):
        self.config = config or RetrievalConfig()
        self.threshold_manager = SimilarityThresholdManager(self.config.min_similarity_threshold)
        self.ranking_system = ResultRankingSystem()
        self.confidence_calculator = ConfidenceCalculator()
        self.scope_expander = SearchScopeExpander()
    
    def enhanced_search(
        self, 
        query: str, 
        tenant_id: int,
        query_complexity: str = "moderate",
        top_k: int = None,
        retrieval_function: Callable = None
    ) -> Tuple[List[SearchResult], SearchMetrics, str]:
        """
        Perform enhanced semantic search with all improvements.
        
        Args:
            query: The search query
            tenant_id: Tenant identifier
            query_complexity: Query complexity level
            top_k: Number of results to retrieve
            retrieval_function: Function to perform document retrieval
        
        Returns:
            Tuple of (search_results, search_metrics, uncertainty_message)
        """
        
        # Import here to avoid circular imports and AWS initialization issues
        if retrieval_function is None:
            try:
                from rag_model.rag_utils import retrieve_s3_vectors
                retrieval_function = retrieve_s3_vectors
            except ImportError:
                raise ValueError("No retrieval function provided and rag_utils not available")
        
        # Use provided top_k or default from config
        search_top_k = top_k or self.config.base_top_k
        
        # Step 1: Retrieve initial results
        initial_documents = retrieval_function(query, tenant_id, search_top_k)
        original_count = len(initial_documents)
        
        # Step 2: Determine adaptive threshold
        adaptive_threshold, threshold_adjustments = self.threshold_manager.get_adaptive_threshold(
            initial_documents, target_result_count=search_top_k, query_complexity=query_complexity
        )
        
        # Step 3: Filter by threshold
        filtered_documents, filtered_count = self.threshold_manager.filter_by_threshold(
            initial_documents, adaptive_threshold
        )
        
        # Step 4: Rank and score results
        search_results = self.ranking_system.rank_results(
            filtered_documents, query, self.config.diversity_threshold
        )
        
        # Step 5: Check if search scope expansion is needed
        should_expand, expansion_reasons = self.scope_expander.should_expand_search(
            search_results, query
        )
        
        expansion_performed = False
        expansion_strategies = []
        
        if should_expand:
            # Expand search scope
            expanded_documents, expanded_config, strategies_used = self.scope_expander.expand_search_scope(
                query, tenant_id, self.config, expansion_reasons, retrieval_function
            )
            
            if len(expanded_documents) > len(filtered_documents):
                # Re-filter with new threshold
                expanded_filtered, _ = self.threshold_manager.filter_by_threshold(
                    expanded_documents, expanded_config.min_similarity_threshold
                )
                
                # Re-rank expanded results
                expanded_search_results = self.ranking_system.rank_results(
                    expanded_filtered, query, expanded_config.diversity_threshold
                )
                
                # Use expanded results if they're better
                if len(expanded_search_results) > len(search_results):
                    search_results = expanded_search_results
                    filtered_documents = expanded_filtered
                    adaptive_threshold = expanded_config.min_similarity_threshold
                    expansion_performed = True
                    expansion_strategies = strategies_used
        
        # Step 6: Calculate overall confidence
        overall_confidence, uncertainty_indicators = self.confidence_calculator.calculate_search_confidence(
            search_results, query, original_count
        )
        
        # Step 7: Generate uncertainty message with alternative suggestions
        uncertainty_message = self.confidence_calculator.generate_uncertainty_message(
            uncertainty_indicators, overall_confidence
        )
        
        # Add alternative query suggestions if confidence is low
        if overall_confidence < 0.6:
            alternatives = self.scope_expander.suggest_alternative_queries(query)
            if alternatives:
                uncertainty_message += f" You might try asking: {', '.join(alternatives[:2])}"
        
        # Step 8: Compile metrics
        search_metrics = SearchMetrics(
            total_results=original_count,
            filtered_results=len(filtered_documents),
            average_similarity=sum(r.similarity_score for r in search_results) / len(search_results) if search_results else 0.0,
            average_confidence=overall_confidence,
            diversity_index=sum(r.diversity_score for r in search_results) / len(search_results) if search_results else 0.0,
            threshold_used=adaptive_threshold,
            adaptive_adjustments=threshold_adjustments + (expansion_strategies if expansion_performed else [])
        )
        
        return search_results, search_metrics, uncertainty_message


# Integration utilities for existing RAG system
def integrate_enhanced_search_with_rag(
    query: str, 
    tenant_id: int, 
    config: RetrievalConfig = None,
    query_complexity: str = "moderate",
    retrieval_function: Callable = None
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Integration function to use enhanced semantic search with existing RAG system.
    Returns documents in the format expected by existing RAG chains.
    """
    
    if retrieval_function is None:
        try:
            from rag_model.rag_utils import retrieve_s3_vectors
            retrieval_function = retrieve_s3_vectors
        except ImportError:
            raise ValueError("No retrieval function provided and rag_utils not available")
    
    enhanced_search = EnhancedSemanticSearch(config)
    search_results, metrics, uncertainty_message = enhanced_search.enhanced_search(
        query, tenant_id, query_complexity, retrieval_function=retrieval_function
    )
    
    # Convert SearchResult objects back to Document objects for compatibility
    documents = [result.document for result in search_results]
    
    # Add enhanced metadata to documents
    for i, (doc, result) in enumerate(zip(documents, search_results)):
        doc.metadata.update({
            'enhanced_rank': result.rank,
            'relevance_score': result.relevance_score,
            'recency_score': result.recency_score,
            'diversity_score': result.diversity_score,
            'confidence_level': result.confidence_level.value,
            'uncertainty_indicators': result.uncertainty_indicators
        })
    
    # Compile integration metadata
    integration_metadata = {
        'search_metrics': {
            'total_results': metrics.total_results,
            'filtered_results': metrics.filtered_results,
            'average_similarity': metrics.average_similarity,
            'average_confidence': metrics.average_confidence,
            'diversity_index': metrics.diversity_index,
            'threshold_used': metrics.threshold_used,
            'adaptive_adjustments': metrics.adaptive_adjustments
        },
        'uncertainty_message': uncertainty_message,
        'enhanced_search_used': True
    }
    
    return documents, integration_metadata


def create_enhanced_retriever(tenant_id: int, config: RetrievalConfig = None):
    """
    Create an enhanced retriever that can be used as a drop-in replacement 
    for the existing S3VectorRetriever.
    """
    
    class EnhancedS3VectorRetriever:
        def __init__(self, tenant_id: int, config: RetrievalConfig = None):
            self.tenant_id = tenant_id
            self.config = config or RetrievalConfig()
            self.enhanced_search = EnhancedSemanticSearch(self.config)
        
        def _get_relevant_documents(self, query: str) -> List[Document]:
            documents, _ = integrate_enhanced_search_with_rag(
                query, self.tenant_id, self.config
            )
            return documents
        
        async def _aget_relevant_documents(self, query: str) -> List[Document]:
            return self._get_relevant_documents(query)
    
    return EnhancedS3VectorRetriever(tenant_id, config)


def enhance_existing_rag_response(
    response_dict: Dict[str, Any], 
    integration_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhance existing RAG response with confidence and uncertainty information.
    """
    
    enhanced_response = response_dict.copy()
    
    # Add confidence information
    if 'search_metrics' in integration_metadata:
        metrics = integration_metadata['search_metrics']
        enhanced_response['search_confidence'] = metrics['average_confidence']
        enhanced_response['search_quality'] = {
            'total_results_found': metrics['total_results'],
            'results_after_filtering': metrics['filtered_results'],
            'average_similarity': metrics['average_similarity'],
            'diversity_score': metrics['diversity_index'],
            'threshold_used': metrics['threshold_used']
        }
    
    # Add uncertainty message if present
    uncertainty_msg = integration_metadata.get('uncertainty_message', '')
    if uncertainty_msg:
        # Append uncertainty message to the answer
        current_answer = enhanced_response.get('answer', '')
        enhanced_response['answer'] = f"{current_answer}\n\n{uncertainty_msg}".strip()
        enhanced_response['uncertainty_indicated'] = True
    else:
        enhanced_response['uncertainty_indicated'] = False
    
    return enhanced_response