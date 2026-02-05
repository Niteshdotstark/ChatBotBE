"""
Dynamic Retrieval System for RAG Chatbot
Implements query complexity analysis, adaptive parameter adjustment, and context-aware retrieval.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from rag_model.config_models import RetrievalConfig, QueryComplexity
from rag_model.context_manager import ConversationContext


class QueryType(Enum):
    """Types of queries based on intent"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"


class QueryAmbiguity(Enum):
    """Levels of query ambiguity"""
    CLEAR = "clear"
    MODERATE = "moderate"
    AMBIGUOUS = "ambiguous"
    HIGHLY_AMBIGUOUS = "highly_ambiguous"


@dataclass
class QueryAnalysis:
    """Results of query complexity analysis"""
    complexity: QueryComplexity
    query_type: QueryType
    ambiguity: QueryAmbiguity
    intent_confidence: float
    key_concepts: List[str]
    requires_context: bool
    estimated_answer_length: str  # "short", "medium", "long"
    follow_up_indicators: List[str]


class QueryComplexityAnalyzer:
    """
    Analyzes query complexity for dynamic retrieval parameter adjustment.
    Implements Requirements 4.1, 4.2 for query classification and intent detection.
    """
    
    def __init__(self):
        self._factual_indicators = [
            "what is", "who is", "when did", "where is", "how many", "which",
            "define", "definition", "meaning", "explain what"
        ]
        
        self._analytical_indicators = [
            "why", "how does", "analyze", "compare", "contrast", "evaluate",
            "assess", "relationship", "impact", "effect", "cause", "reason"
        ]
        
        self._procedural_indicators = [
            "how to", "steps", "process", "procedure", "method", "way to",
            "guide", "tutorial", "instructions", "implement"
        ]
        
        self._comparative_indicators = [
            "difference", "similar", "versus", "vs", "better", "worse",
            "compare", "contrast", "alternative", "option"
        ]
        
        self._exploratory_indicators = [
            "explore", "discover", "find out", "learn about", "tell me about",
            "overview", "introduction", "background", "context"
        ]
        
        self._ambiguity_indicators = [
            "something", "anything", "everything", "stuff", "things", "it",
            "this", "that", "those", "these", "some", "any"
        ]
        
        self._follow_up_indicators = [
            "also", "additionally", "furthermore", "moreover", "besides",
            "what about", "how about", "and", "plus", "as well"
        ]
        
        self._technical_terms = [
            "api", "function", "method", "class", "variable", "parameter",
            "algorithm", "database", "server", "client", "framework",
            "library", "module", "component", "interface", "protocol"
        ]
    
    def analyze_query_complexity(self, query: str, context: Optional[ConversationContext] = None) -> QueryAnalysis:
        """
        Analyze query complexity and characteristics.
        Implements Requirements 4.1, 4.2 for query classification and intent detection.
        """
        query_lower = query.lower().strip()
        
        # Determine query type and complexity
        query_type = self._classify_query_type(query_lower)
        complexity = self._calculate_complexity_score(query_lower, query_type, context)
        ambiguity = self._assess_ambiguity(query_lower, context)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(query_lower)
        
        # Determine if context is required
        requires_context = self._requires_conversation_context(query_lower, context)
        
        # Estimate answer length needed
        estimated_length = self._estimate_answer_length(query_type, complexity, len(query.split()))
        
        # Detect follow-up indicators
        follow_up_indicators = self._detect_follow_up_indicators(query_lower)
        
        # Calculate intent confidence
        intent_confidence = self._calculate_intent_confidence(query_type, ambiguity, key_concepts)
        
        return QueryAnalysis(
            complexity=complexity,
            query_type=query_type,
            ambiguity=ambiguity,
            intent_confidence=intent_confidence,
            key_concepts=key_concepts,
            requires_context=requires_context,
            estimated_answer_length=estimated_length,
            follow_up_indicators=follow_up_indicators
        )
    
    def _classify_query_type(self, query_lower: str) -> QueryType:
        """Classify the type of query based on linguistic patterns"""
        
        # Count indicators for each type
        type_scores = {
            QueryType.FACTUAL: sum(1 for indicator in self._factual_indicators if indicator in query_lower),
            QueryType.ANALYTICAL: sum(1 for indicator in self._analytical_indicators if indicator in query_lower),
            QueryType.PROCEDURAL: sum(1 for indicator in self._procedural_indicators if indicator in query_lower),
            QueryType.COMPARATIVE: sum(1 for indicator in self._comparative_indicators if indicator in query_lower),
            QueryType.EXPLORATORY: sum(1 for indicator in self._exploratory_indicators if indicator in query_lower)
        }
        
        # Add pattern-based scoring
        if re.search(r'\b(what|who|when|where)\s+(is|are|was|were)\b', query_lower):
            type_scores[QueryType.FACTUAL] += 2
        
        if re.search(r'\b(why|how)\s+(does|do|did|can|could|would|should)\b', query_lower):
            type_scores[QueryType.ANALYTICAL] += 2
        
        if re.search(r'\b(how\s+to|step\s+by\s+step|guide\s+to)\b', query_lower):
            type_scores[QueryType.PROCEDURAL] += 3
        
        if re.search(r'\b(compare|contrast|difference|versus|vs|better|worse)\b', query_lower):
            type_scores[QueryType.COMPARATIVE] += 2
        
        if re.search(r'\b(tell\s+me\s+about|overview|introduction|background)\b', query_lower):
            type_scores[QueryType.EXPLORATORY] += 2
        
        # Return type with highest score, default to factual
        max_type = max(type_scores.items(), key=lambda x: x[1])
        return max_type[0] if max_type[1] > 0 else QueryType.FACTUAL
    
    def _calculate_complexity_score(
        self, 
        query_lower: str, 
        query_type: QueryType, 
        context: Optional[ConversationContext]
    ) -> QueryComplexity:
        """Calculate overall query complexity"""
        
        complexity_score = 0
        
        # Base complexity by type
        type_complexity = {
            QueryType.FACTUAL: 1,
            QueryType.EXPLORATORY: 2,
            QueryType.PROCEDURAL: 3,
            QueryType.COMPARATIVE: 4,
            QueryType.ANALYTICAL: 5
        }
        complexity_score += type_complexity[query_type]
        
        # Length complexity
        word_count = len(query_lower.split())
        if word_count > 20:
            complexity_score += 3
        elif word_count > 10:
            complexity_score += 2
        elif word_count > 5:
            complexity_score += 1
        
        # Multiple questions or concepts
        question_count = query_lower.count('?') + len(re.findall(r'\b(and|or|also)\b', query_lower))
        complexity_score += min(question_count, 3)
        
        # Technical terminology
        technical_count = sum(1 for term in self._technical_terms if term in query_lower)
        complexity_score += min(technical_count, 2)
        
        # Context dependency
        if context and self._requires_conversation_context(query_lower, context):
            complexity_score += 2
        
        # Conditional or hypothetical language
        if re.search(r'\b(if|suppose|assume|hypothetically|what if|would|could|should)\b', query_lower):
            complexity_score += 2
        
        # Map score to complexity enum
        if complexity_score >= 12:
            return QueryComplexity.ANALYTICAL
        elif complexity_score >= 8:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 4:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _assess_ambiguity(self, query_lower: str, context: Optional[ConversationContext]) -> QueryAmbiguity:
        """Assess query ambiguity level"""
        
        ambiguity_score = 0
        
        # Pronoun and vague reference count
        ambiguous_terms = sum(1 for term in self._ambiguity_indicators if term in query_lower)
        ambiguity_score += ambiguous_terms * 2
        
        # Missing key information
        if len(query_lower.split()) < 3:
            ambiguity_score += 2
        
        # Vague verbs
        vague_verbs = ["do", "make", "get", "have", "use", "work", "handle"]
        vague_verb_count = sum(1 for verb in vague_verbs if f" {verb} " in f" {query_lower} ")
        ambiguity_score += vague_verb_count
        
        # Context dependency without context
        if not context and any(term in query_lower for term in ["this", "that", "it", "they"]):
            ambiguity_score += 3
        
        # Multiple possible interpretations
        if re.search(r'\b(or|either|maybe|perhaps|possibly)\b', query_lower):
            ambiguity_score += 1
        
        # Map score to ambiguity enum
        if ambiguity_score >= 8:
            return QueryAmbiguity.HIGHLY_AMBIGUOUS
        elif ambiguity_score >= 5:
            return QueryAmbiguity.AMBIGUOUS
        elif ambiguity_score >= 2:
            return QueryAmbiguity.MODERATE
        else:
            return QueryAmbiguity.CLEAR
    
    def _extract_key_concepts(self, query_lower: str) -> List[str]:
        """Extract key concepts from the query"""
        
        # Remove stop words and common question words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "what", "how", "why", "when", "where", "which", "who", "can", "could",
            "would", "should", "do", "does", "did", "have", "has", "had"
        }
        
        words = query_lower.split()
        key_concepts = []
        
        # Extract meaningful words
        for word in words:
            # Clean word
            clean_word = re.sub(r'[^\w]', '', word)
            
            if (len(clean_word) > 2 and 
                clean_word not in stop_words and 
                not clean_word.isdigit()):
                key_concepts.append(clean_word)
        
        # Extract noun phrases (simple approach)
        noun_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query_lower.title())
        key_concepts.extend([phrase.lower() for phrase in noun_phrases])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in key_concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)
        
        return unique_concepts[:10]  # Limit to top 10 concepts
    
    def _requires_conversation_context(self, query_lower: str, context: Optional[ConversationContext]) -> bool:
        """Determine if query requires conversation context"""
        
        # Explicit context references
        context_indicators = [
            "this", "that", "it", "they", "them", "these", "those",
            "above", "below", "previous", "earlier", "before", "after",
            "also", "too", "as well", "additionally", "furthermore"
        ]
        
        if any(indicator in query_lower for indicator in context_indicators):
            return True
        
        # Follow-up question patterns
        if any(indicator in query_lower for indicator in self._follow_up_indicators):
            return True
        
        # Short queries often need context
        if len(query_lower.split()) < 4 and context:
            return True
        
        # Questions without clear subject
        if query_lower.startswith(("how", "why", "what")) and "?" in query_lower:
            # Check if there's a clear subject
            if not any(concept in query_lower for concept in ["about", "is", "are", "does", "do"]):
                return True
        
        return False
    
    def _estimate_answer_length(self, query_type: QueryType, complexity: QueryComplexity, query_word_count: int) -> str:
        """Estimate the expected length of the answer"""
        
        # Base length by query type
        type_lengths = {
            QueryType.FACTUAL: "short",
            QueryType.EXPLORATORY: "medium",
            QueryType.PROCEDURAL: "long",
            QueryType.COMPARATIVE: "medium",
            QueryType.ANALYTICAL: "long"
        }
        
        base_length = type_lengths[query_type]
        
        # Adjust based on complexity
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.ANALYTICAL]:
            if base_length == "short":
                return "medium"
            elif base_length == "medium":
                return "long"
        
        # Adjust based on query length
        if query_word_count > 15:
            if base_length == "short":
                return "medium"
        
        return base_length
    
    def _detect_follow_up_indicators(self, query_lower: str) -> List[str]:
        """Detect indicators that this is a follow-up question"""
        
        detected_indicators = []
        
        for indicator in self._follow_up_indicators:
            if indicator in query_lower:
                detected_indicators.append(indicator)
        
        # Pattern-based detection
        if re.search(r'\b(what about|how about|and what|and how)\b', query_lower):
            detected_indicators.append("follow_up_pattern")
        
        if query_lower.startswith(("and", "also", "plus", "additionally")):
            detected_indicators.append("continuation_start")
        
        return detected_indicators
    
    def _calculate_intent_confidence(
        self, 
        query_type: QueryType, 
        ambiguity: QueryAmbiguity, 
        key_concepts: List[str]
    ) -> float:
        """Calculate confidence in intent detection"""
        
        base_confidence = 0.7
        
        # Adjust based on ambiguity
        ambiguity_adjustments = {
            QueryAmbiguity.CLEAR: 0.2,
            QueryAmbiguity.MODERATE: 0.0,
            QueryAmbiguity.AMBIGUOUS: -0.2,
            QueryAmbiguity.HIGHLY_AMBIGUOUS: -0.4
        }
        
        confidence = base_confidence + ambiguity_adjustments[ambiguity]
        
        # Adjust based on key concepts
        if len(key_concepts) >= 3:
            confidence += 0.1
        elif len(key_concepts) <= 1:
            confidence -= 0.1
        
        # Ensure confidence is within bounds
        return max(0.1, min(1.0, confidence))


class AdaptiveParameterAdjuster:
    """
    Adjusts retrieval parameters based on query complexity and context.
    Implements for dynamic parameter adjustment.
    """
    
    def __init__(self, base_config: RetrievalConfig):
        self.base_config = base_config
    
    def adjust_retrieval_parameters(
        self, 
        query_analysis: QueryAnalysis, 
        context: Optional[ConversationContext] = None
    ) -> RetrievalConfig:
        """
        Adjust retrieval parameters based on query analysis.
        Implements Requirements dynamic top_k adjustment and confidence-based expansion.
        """
        
        # Start with base configuration
        adjusted_config = RetrievalConfig(
            base_top_k=self.base_config.base_top_k,
            max_top_k=self.base_config.max_top_k,
            min_similarity_threshold=self.base_config.min_similarity_threshold,
            diversity_threshold=self.base_config.diversity_threshold,
            rerank_enabled=self.base_config.rerank_enabled,
            hybrid_search_weight=self.base_config.hybrid_search_weight
        )
        
        # Adjust top_k based on complexity
        complexity_multipliers = {
            QueryComplexity.SIMPLE: 0.7,
            QueryComplexity.MODERATE: 1.0,
            QueryComplexity.COMPLEX: 1.4,
            QueryComplexity.ANALYTICAL: 1.8
        }
        
        multiplier = complexity_multipliers[query_analysis.complexity]
        adjusted_top_k = int(self.base_config.base_top_k * multiplier)
        adjusted_config.base_top_k = min(adjusted_top_k, self.base_config.max_top_k)
        
        # Adjust based on query type
        if query_analysis.query_type == QueryType.COMPARATIVE:
            # Need more diverse results for comparisons
            adjusted_config.base_top_k = min(adjusted_config.base_top_k + 3, self.base_config.max_top_k)
            adjusted_config.diversity_threshold = max(0.9, adjusted_config.diversity_threshold)
        
        elif query_analysis.query_type == QueryType.PROCEDURAL:
            # Need comprehensive step-by-step information
            adjusted_config.base_top_k = min(adjusted_config.base_top_k + 2, self.base_config.max_top_k)
        
        elif query_analysis.query_type == QueryType.FACTUAL and query_analysis.complexity == QueryComplexity.SIMPLE:
            # Simple factual queries need fewer, more precise results
            adjusted_config.base_top_k = max(3, adjusted_config.base_top_k - 2)
            adjusted_config.min_similarity_threshold = min(0.85, adjusted_config.min_similarity_threshold + 0.1)
        
        # Adjust based on ambiguity
        if query_analysis.ambiguity in [QueryAmbiguity.AMBIGUOUS, QueryAmbiguity.HIGHLY_AMBIGUOUS]:
            # More diverse results for ambiguous queries
            adjusted_config.base_top_k = min(adjusted_config.base_top_k + 2, self.base_config.max_top_k)
            adjusted_config.diversity_threshold = max(0.85, adjusted_config.diversity_threshold)
            adjusted_config.min_similarity_threshold = max(0.6, adjusted_config.min_similarity_threshold - 0.1)
        
        # Adjust based on intent confidence
        if query_analysis.intent_confidence < 0.6:
            # Low confidence - cast wider net
            adjusted_config.base_top_k = min(adjusted_config.base_top_k + 2, self.base_config.max_top_k)
            adjusted_config.min_similarity_threshold = max(0.65, adjusted_config.min_similarity_threshold - 0.05)
        
        elif query_analysis.intent_confidence > 0.9:
            # High confidence - can be more selective
            adjusted_config.min_similarity_threshold = min(0.8, adjusted_config.min_similarity_threshold + 0.05)
        
        # Adjust based on estimated answer length
        if query_analysis.estimated_answer_length == "long":
            adjusted_config.base_top_k = min(adjusted_config.base_top_k + 3, self.base_config.max_top_k)
        elif query_analysis.estimated_answer_length == "short":
            adjusted_config.base_top_k = max(3, adjusted_config.base_top_k - 1)
        
        # Context-based adjustments
        if context:
            if query_analysis.requires_context:
                # Follow-up questions might need broader search
                adjusted_config.base_top_k = min(adjusted_config.base_top_k + 1, self.base_config.max_top_k)
            
            # Adjust based on conversation complexity
            if hasattr(context, 'complexity'):
                if context.complexity.value in ['complex', 'analytical']:
                    adjusted_config.base_top_k = min(adjusted_config.base_top_k + 2, self.base_config.max_top_k)
        
        # Enable reranking for complex queries
        if query_analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.ANALYTICAL]:
            adjusted_config.rerank_enabled = True
        
        return adjusted_config
    
    def should_expand_retrieval(
        self, 
        initial_results: List[Document], 
        query_analysis: QueryAnalysis,
        confidence_threshold: float = 0.7
    ) -> bool:
        """
        Determine if retrieval should be expanded based on initial results confidence.
        Implements Requirements 4.4 for confidence-based retrieval expansion.
        """
        
        if not initial_results:
            return True
        
        # Calculate average confidence from similarity scores
        total_confidence = 0.0
        confidence_count = 0
        
        for doc in initial_results:
            if hasattr(doc, 'metadata') and 'similarity_score' in doc.metadata:
                total_confidence += doc.metadata['similarity_score']
                confidence_count += 1
        
        if confidence_count == 0:
            # No confidence scores available, use query characteristics
            return query_analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.ANALYTICAL]
        
        avg_confidence = total_confidence / confidence_count
        
        # Adjust threshold based on query characteristics
        adjusted_threshold = confidence_threshold
        
        if query_analysis.ambiguity in [QueryAmbiguity.AMBIGUOUS, QueryAmbiguity.HIGHLY_AMBIGUOUS]:
            adjusted_threshold -= 0.1
        
        if query_analysis.intent_confidence < 0.6:
            adjusted_threshold -= 0.1
        
        if query_analysis.query_type == QueryType.COMPARATIVE:
            adjusted_threshold -= 0.05  # Comparisons often need more sources
        
        return avg_confidence < adjusted_threshold


class ContextAwareRetriever:
    """
    Integrates conversation context into retrieval decisions and handles follow-up questions.
    Implements Requirements 4.5 for context-aware retrieval and follow-up question handling.
    """
    
    def __init__(self, base_config: RetrievalConfig):
        self.base_config = base_config
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.parameter_adjuster = AdaptiveParameterAdjuster(base_config)
    
    def enhance_query_with_context(
        self, 
        query: str, 
        context: Optional[ConversationContext]
    ) -> Tuple[str, List[str]]:
        """
        Enhance query with conversation context for better retrieval.
        Implements Requirements 4.5 for context integration into retrieval decisions.
        """
        
        if not context or not context.recent_messages:
            return query, []
        
        enhanced_terms = []
        context_keywords = []
        
        # Extract context from recent messages
        recent_topics = set()
        recent_entities = set()
        
        for message in context.recent_messages[-6:]:  # Last 3 exchanges
            if hasattr(message, 'content'):
                content = message.content.lower()
                
                # Extract potential entities (capitalized words)
                entities = re.findall(r'\b[A-Z][a-z]+\b', message.content)
                recent_entities.update([entity.lower() for entity in entities])
                
                # Extract key terms from context
                words = content.split()
                meaningful_words = [
                    word for word in words 
                    if len(word) > 3 and word not in {
                        "what", "how", "why", "when", "where", "which", "who",
                        "the", "and", "or", "but", "with", "for", "from"
                    }
                ]
                recent_topics.update(meaningful_words[:5])  # Top 5 per message
        
        # Add context topics if query is ambiguous or uses pronouns
        query_lower = query.lower()
        needs_context_expansion = (
            any(pronoun in query_lower for pronoun in ["it", "this", "that", "they", "them"]) or
            len(query.split()) < 5 or
            any(word in query_lower for word in ["also", "too", "as well", "additionally"])
        )
        
        if needs_context_expansion:
            # Add relevant context terms
            for topic in list(recent_topics)[:3]:  # Top 3 context terms
                if topic not in query_lower:
                    enhanced_terms.append(topic)
                    context_keywords.append(topic)
            
            for entity in list(recent_entities)[:2]:  # Top 2 entities
                if entity not in query_lower:
                    enhanced_terms.append(entity)
                    context_keywords.append(entity)
        
        # Create enhanced query
        if enhanced_terms:
            enhanced_query = f"{query} {' '.join(enhanced_terms)}"
        else:
            enhanced_query = query
        
        return enhanced_query, context_keywords
    
    def handle_follow_up_question(
        self, 
        query: str, 
        context: ConversationContext,
        query_analysis: QueryAnalysis
    ) -> Dict[str, Any]:
        """
        Handle follow-up questions by analyzing conversation flow.
        Implements Requirements 4.5 for follow-up question handling.
        """
        
        follow_up_info = {
            "is_follow_up": len(query_analysis.follow_up_indicators) > 0,
            "reference_context": [],
            "topic_continuation": False,
            "clarification_request": False,
            "expansion_request": False
        }
        
        if not follow_up_info["is_follow_up"]:
            return follow_up_info
        
        # Analyze the type of follow-up
        query_lower = query.lower()
        
        # Check for clarification requests
        clarification_patterns = [
            "what do you mean", "can you explain", "clarify", "elaborate",
            "more details", "be more specific", "what exactly"
        ]
        
        if any(pattern in query_lower for pattern in clarification_patterns):
            follow_up_info["clarification_request"] = True
            # Reference the last AI response for clarification
            if context.recent_messages:
                for i in range(len(context.recent_messages) - 1, -1, -1):
                    if isinstance(context.recent_messages[i], AIMessage):
                        follow_up_info["reference_context"].append({
                            "type": "previous_response",
                            "content": context.recent_messages[i].content[:200],
                            "index": i
                        })
                        break
        
        # Check for expansion requests
        expansion_patterns = [
            "tell me more", "what else", "anything else", "more information",
            "expand on", "go deeper", "additional"
        ]
        
        if any(pattern in query_lower for pattern in expansion_patterns):
            follow_up_info["expansion_request"] = True
        
        # Check for topic continuation
        if any(indicator in query_lower for indicator in ["also", "and", "plus", "additionally"]):
            follow_up_info["topic_continuation"] = True
            
            # Extract the main topic from recent conversation
            if context.topic_context:
                follow_up_info["reference_context"].append({
                    "type": "topic_context",
                    "content": context.topic_context[:3],  # Top 3 topics
                    "index": -1
                })
        
        # Extract specific references
        reference_patterns = [
            r"the (\w+) you mentioned",
            r"that (\w+)",
            r"this (\w+)",
            r"those (\w+)",
            r"these (\w+)"
        ]
        
        for pattern in reference_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                follow_up_info["reference_context"].append({
                    "type": "specific_reference",
                    "content": match,
                    "index": -1
                })
        
        return follow_up_info
    
    def adjust_retrieval_for_context(
        self, 
        base_config: RetrievalConfig,
        context: ConversationContext,
        follow_up_info: Dict[str, Any]
    ) -> RetrievalConfig:
        """
        Adjust retrieval parameters based on conversation context.
        """
        
        adjusted_config = RetrievalConfig(
            base_top_k=base_config.base_top_k,
            max_top_k=base_config.max_top_k,
            min_similarity_threshold=base_config.min_similarity_threshold,
            diversity_threshold=base_config.diversity_threshold,
            rerank_enabled=base_config.rerank_enabled,
            hybrid_search_weight=base_config.hybrid_search_weight
        )
        
        # Adjust for follow-up questions
        if follow_up_info["is_follow_up"]:
            if follow_up_info["clarification_request"]:
                # Clarifications need fewer, more precise results
                adjusted_config.base_top_k = max(3, adjusted_config.base_top_k - 2)
                adjusted_config.min_similarity_threshold = min(0.8, adjusted_config.min_similarity_threshold + 0.05)
            
            elif follow_up_info["expansion_request"]:
                # Expansions need more diverse results
                adjusted_config.base_top_k = min(adjusted_config.base_top_k + 3, adjusted_config.max_top_k)
                adjusted_config.diversity_threshold = max(0.85, adjusted_config.diversity_threshold)
            
            elif follow_up_info["topic_continuation"]:
                # Topic continuations need moderate expansion
                adjusted_config.base_top_k = min(adjusted_config.base_top_k + 1, adjusted_config.max_top_k)
        
        # Adjust based on conversation complexity
        if hasattr(context, 'complexity'):
            complexity_adjustments = {
                'simple': 0,
                'moderate': 1,
                'complex': 2,
                'analytical': 3
            }
            
            complexity_value = getattr(context.complexity, 'value', 'moderate')
            adjustment = complexity_adjustments.get(complexity_value, 1)
            adjusted_config.base_top_k = min(
                adjusted_config.base_top_k + adjustment, 
                adjusted_config.max_top_k
            )
        
        # Adjust based on context utilization
        if context.relevance_scores:
            avg_relevance = sum(context.relevance_scores.values()) / len(context.relevance_scores)
            if avg_relevance < 0.5:
                # Low context relevance - might need broader search
                adjusted_config.min_similarity_threshold = max(
                    0.6, adjusted_config.min_similarity_threshold - 0.05
                )
        
        return adjusted_config


class DynamicRetriever:
    """
    Main dynamic retrieval system that combines query analysis, parameter adjustment, and context awareness.
    Implements the complete dynamic retrieval system from the design document.
    """
    
    def __init__(self, base_config: RetrievalConfig = None):
        self.base_config = base_config or RetrievalConfig()
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.parameter_adjuster = AdaptiveParameterAdjuster(self.base_config)
        self.context_retriever = ContextAwareRetriever(self.base_config)
    
    def analyze_and_retrieve(
        self, 
        query: str, 
        tenant_id: int,
        context: Optional[ConversationContext] = None,
        retrieval_function: callable = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Complete dynamic retrieval process with analysis, parameter adjustment, and context integration.
        
        Args:
            query: The user's query
            tenant_id: Tenant identifier for document retrieval
            context: Conversation context
            retrieval_function: Function to perform actual document retrieval
        
        Returns:
            Tuple of (retrieved_documents, retrieval_metadata)
        """
        
        # Step 1: Analyze query complexity
        query_analysis = self.complexity_analyzer.analyze_query_complexity(query, context)
        
        # Step 2: Enhance query with context if needed
        enhanced_query, context_keywords = self.context_retriever.enhance_query_with_context(query, context)
        
        # Step 3: Handle follow-up questions
        follow_up_info = {}
        if context:
            follow_up_info = self.context_retriever.handle_follow_up_question(query, context, query_analysis)
        
        # Step 4: Adjust retrieval parameters
        adjusted_config = self.parameter_adjuster.adjust_retrieval_parameters(query_analysis, context)
        
        # Step 5: Further adjust for context if available
        if context:
            adjusted_config = self.context_retriever.adjust_retrieval_for_context(
                adjusted_config, context, follow_up_info
            )
        
        # Step 6: Perform retrieval (use provided function or default)
        if retrieval_function:
            initial_results = retrieval_function(enhanced_query, tenant_id, adjusted_config.base_top_k)
        else:
            # Fallback to basic retrieval if no function provided
            from rag_model.rag_utils import retrieve_s3_vectors
            initial_results = retrieve_s3_vectors(enhanced_query, tenant_id, adjusted_config.base_top_k)
        
        # Step 7: Check if retrieval expansion is needed
        should_expand = self.parameter_adjuster.should_expand_retrieval(initial_results, query_analysis)
        
        final_results = initial_results
        expansion_performed = False
        
        if should_expand and adjusted_config.base_top_k < adjusted_config.max_top_k:
            # Expand retrieval with higher top_k
            expanded_top_k = min(adjusted_config.base_top_k + 5, adjusted_config.max_top_k)
            
            if retrieval_function:
                expanded_results = retrieval_function(enhanced_query, tenant_id, expanded_top_k)
            else:
                from rag_model.rag_utils import retrieve_s3_vectors
                expanded_results = retrieve_s3_vectors(enhanced_query, tenant_id, expanded_top_k)
            
            final_results = expanded_results
            expansion_performed = True
        
        # Step 8: Compile retrieval metadata
        retrieval_metadata = {
            "query_analysis": {
                "complexity": query_analysis.complexity.value,
                "query_type": query_analysis.query_type.value,
                "ambiguity": query_analysis.ambiguity.value,
                "intent_confidence": query_analysis.intent_confidence,
                "key_concepts": query_analysis.key_concepts,
                "requires_context": query_analysis.requires_context,
                "estimated_answer_length": query_analysis.estimated_answer_length
            },
            "retrieval_config": {
                "top_k_used": adjusted_config.base_top_k,
                "similarity_threshold": adjusted_config.min_similarity_threshold,
                "diversity_threshold": adjusted_config.diversity_threshold,
                "rerank_enabled": adjusted_config.rerank_enabled
            },
            "context_enhancement": {
                "query_enhanced": enhanced_query != query,
                "context_keywords": context_keywords,
                "follow_up_info": follow_up_info
            },
            "retrieval_performance": {
                "initial_result_count": len(initial_results),
                "final_result_count": len(final_results),
                "expansion_performed": expansion_performed,
                "expansion_triggered": should_expand
            }
        }
        
        return final_results, retrieval_metadata