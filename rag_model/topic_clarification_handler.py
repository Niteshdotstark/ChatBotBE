"""
Advanced Topic Shift and Clarification Handling for RAG Chatbot
Implements topic change detection, retrieval adaptation, and reference to previous responses.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from rag_model.context_manager import ConversationContext
from rag_model.dynamic_retriever import QueryAnalysis, DynamicRetriever
from rag_model.config_models import RetrievalConfig


class ClarificationType(Enum):
    """Types of clarification requests"""
    DEFINITION_REQUEST = "definition_request"
    ELABORATION_REQUEST = "elaboration_request"
    EXAMPLE_REQUEST = "example_request"
    PROCESS_CLARIFICATION = "process_clarification"
    COMPARISON_REQUEST = "comparison_request"
    CONTEXT_CLARIFICATION = "context_clarification"


class TopicShiftType(Enum):
    """Enhanced types of topic shifts"""
    NO_SHIFT = "no_shift"
    GRADUAL_EVOLUTION = "gradual_evolution"
    RELATED_BRANCH = "related_branch"
    ABRUPT_CHANGE = "abrupt_change"
    RETURN_TO_PREVIOUS = "return_to_previous"
    MULTI_TOPIC_QUERY = "multi_topic_query"


@dataclass
class ClarificationRequest:
    """Detailed clarification request analysis"""
    is_clarification: bool
    clarification_type: ClarificationType
    target_concept: str
    reference_message_index: int
    specific_phrase: str
    confidence: float
    suggested_response_approach: str


@dataclass
class EnhancedTopicShift:
    """Enhanced topic shift detection with retrieval adaptation"""
    shift_type: TopicShiftType
    confidence: float
    previous_topics: List[str]
    current_topics: List[str]
    topic_similarity_score: float
    retrieval_adaptation_needed: bool
    suggested_retrieval_strategy: str
    context_preservation_level: float


@dataclass
class RetrievalAdaptation:
    """Retrieval strategy adaptation for topic shifts"""
    expand_search_scope: bool
    include_previous_context: bool
    adjust_similarity_threshold: float
    modify_top_k: int
    add_topic_keywords: List[str]
    prioritize_recent_docs: bool


class AdvancedClarificationDetector:
    """
    Advanced clarification request detection and handling.
    Implements Requirements 6.4 for reference to previous responses for clarifications.
    """
    
    def __init__(self):
        self._clarification_patterns = {
            ClarificationType.DEFINITION_REQUEST: [
                r"what (?:do you mean by|is) (.+?)\?",
                r"(?:define|explain) (.+?)(?:\?|$)",
                r"what does (.+?) mean",
                r"(?:can you )?(?:clarify|explain) (?:what )?(.+?) (?:is|means?)"
            ],
            ClarificationType.ELABORATION_REQUEST: [
                r"(?:can you )?(?:elaborate|expand) (?:on )?(.+?)\?",
                r"tell me more about (.+?)(?:\?|$)",
                r"(?:more )?(?:details?|information) (?:about|on) (.+?)(?:\?|$)",
                r"(?:can you )?(?:go )?(?:deeper|further) (?:into )?(.+?)\?"
            ],
            ClarificationType.EXAMPLE_REQUEST: [
                r"(?:can you )?(?:give|provide) (?:me )?(?:an? )?example (?:of )?(.+?)\?",
                r"what (?:are )?(?:some )?examples? (?:of )?(.+?)\?",
                r"(?:show me|demonstrate) (.+?)(?:\?|$)",
                r"for instance, (.+?)\?"
            ],
            ClarificationType.PROCESS_CLARIFICATION: [
                r"how (?:do (?:you|i)|does (?:one|it)) (.+?)\?",
                r"what (?:are )?(?:the )?steps? (?:to|for) (.+?)\?",
                r"(?:walk me through|explain the process of) (.+?)(?:\?|$)",
                r"how (?:exactly )?(?:do|does) (.+?) work\?"
            ],
            ClarificationType.COMPARISON_REQUEST: [
                r"(?:what (?:is|are) )?(?:the )?difference(?:s)? between (.+?) and (.+?)\?",
                r"how (?:does|do) (.+?) (?:compare to|differ from) (.+?)\?",
                r"(.+?) (?:vs|versus) (.+?)\?",
                r"which is better[,:]? (.+?) or (.+?)\?"
            ],
            ClarificationType.CONTEXT_CLARIFICATION: [
                r"what did you mean (?:by|when you said) (.+?)\?",
                r"(?:can you )?clarify (?:your )?(?:previous )?(?:statement|response|answer) (?:about )?(.+?)\?",
                r"i don't understand (?:your )?(?:point about|what you said about) (.+?)(?:\?|$)",
                r"(?:when you said|you mentioned) (.+?)[,.]? (?:what did you mean|can you explain)\?"
            ]
        }
        
        self._reference_indicators = [
            "you said", "you mentioned", "your previous", "earlier you",
            "above", "before", "previously", "that response", "your answer"
        ]
        
        self._clarification_triggers = [
            "clarify", "explain", "elaborate", "what do you mean",
            "i don't understand", "can you", "help me understand",
            "more details", "be more specific"
        ]
    
    def detect_clarification_request(
        self, 
        query: str, 
        context: ConversationContext
    ) -> ClarificationRequest:
        """
        Detect and analyze clarification requests.
        Implements Requirements 6.4 for clarification request detection.
        """
        
        query_lower = query.lower().strip()
        
        # Check if this is a clarification request
        is_clarification = any(trigger in query_lower for trigger in self._clarification_triggers)
        
        if not is_clarification:
            return ClarificationRequest(
                is_clarification=False,
                clarification_type=ClarificationType.DEFINITION_REQUEST,
                target_concept="",
                reference_message_index=-1,
                specific_phrase="",
                confidence=0.0,
                suggested_response_approach="standard_response"
            )
        
        # Determine clarification type and extract target concept
        clarification_type, target_concept, confidence = self._classify_clarification_type(query_lower)
        
        # Find reference to previous message
        reference_index, specific_phrase = self._find_message_reference(query_lower, context)
        
        # Determine response approach
        response_approach = self._determine_response_approach(clarification_type, reference_index)
        
        return ClarificationRequest(
            is_clarification=True,
            clarification_type=clarification_type,
            target_concept=target_concept,
            reference_message_index=reference_index,
            specific_phrase=specific_phrase,
            confidence=confidence,
            suggested_response_approach=response_approach
        )
    
    def _classify_clarification_type(self, query: str) -> Tuple[ClarificationType, str, float]:
        """Classify the type of clarification request"""
        
        best_match = None
        best_confidence = 0.0
        best_concept = ""
        
        for clarif_type, patterns in self._clarification_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    confidence = 0.8  # Base confidence for pattern match
                    
                    # Extract concept from match groups
                    concept = match.group(1) if match.groups() else ""
                    
                    # Adjust confidence based on pattern specificity
                    if len(pattern) > 30:  # More specific patterns get higher confidence
                        confidence += 0.1
                    
                    if confidence > best_confidence:
                        best_match = clarif_type
                        best_confidence = confidence
                        best_concept = concept.strip()
        
        # Default to definition request if no specific pattern matched
        if best_match is None:
            return ClarificationType.DEFINITION_REQUEST, "", 0.3
        
        return best_match, best_concept, best_confidence
    
    def _find_message_reference(
        self, 
        query: str, 
        context: ConversationContext
    ) -> Tuple[int, str]:
        """Find reference to specific previous message"""
        
        if not context.recent_messages:
            return -1, ""
        
        # Check for explicit reference indicators
        for indicator in self._reference_indicators:
            if indicator in query:
                # Find the most recent AI message (likely what user is referring to)
                for i in range(len(context.recent_messages) - 1, -1, -1):
                    if isinstance(context.recent_messages[i], AIMessage):
                        # Extract the phrase being referenced
                        phrase_match = re.search(
                            rf"{indicator}\s+(.+?)(?:\?|$|,|\.|;)", 
                            query, 
                            re.IGNORECASE
                        )
                        phrase = phrase_match.group(1) if phrase_match else ""
                        return i, phrase.strip()
        
        # If no explicit reference, assume referring to last AI response
        for i in range(len(context.recent_messages) - 1, -1, -1):
            if isinstance(context.recent_messages[i], AIMessage):
                return i, ""
        
        return -1, ""
    
    def _determine_response_approach(
        self, 
        clarification_type: ClarificationType, 
        reference_index: int
    ) -> str:
        """Determine the best approach for responding to the clarification"""
        
        approach_mapping = {
            ClarificationType.DEFINITION_REQUEST: "provide_definition_with_context",
            ClarificationType.ELABORATION_REQUEST: "expand_with_details",
            ClarificationType.EXAMPLE_REQUEST: "provide_concrete_examples",
            ClarificationType.PROCESS_CLARIFICATION: "break_down_steps",
            ClarificationType.COMPARISON_REQUEST: "structured_comparison",
            ClarificationType.CONTEXT_CLARIFICATION: "reference_and_clarify"
        }
        
        base_approach = approach_mapping[clarification_type]
        
        # Modify approach based on whether we have a specific reference
        if reference_index >= 0:
            return f"{base_approach}_with_reference"
        else:
            return base_approach


class EnhancedTopicShiftDetector:
    """
    Enhanced topic shift detection with retrieval adaptation.
    Implements Requirements 6.3 for topic change detection and retrieval adaptation.
    """
    
    def __init__(self):
        self._topic_hierarchies = {
            "technology": {
                "programming": ["python", "javascript", "java", "c++", "coding", "development"],
                "data": ["database", "sql", "analytics", "data science", "machine learning"],
                "infrastructure": ["server", "cloud", "aws", "docker", "kubernetes", "devops"],
                "web": ["html", "css", "react", "frontend", "backend", "api"]
            },
            "business": {
                "strategy": ["planning", "goals", "objectives", "vision", "mission"],
                "operations": ["process", "workflow", "efficiency", "optimization"],
                "finance": ["budget", "revenue", "profit", "cost", "roi", "investment"],
                "marketing": ["campaign", "branding", "customer", "market", "sales"]
            },
            "science": {
                "research": ["study", "experiment", "hypothesis", "methodology", "analysis"],
                "mathematics": ["equation", "formula", "calculation", "statistics", "probability"],
                "physics": ["energy", "force", "motion", "quantum", "relativity"],
                "biology": ["cell", "organism", "evolution", "genetics", "ecosystem"]
            }
        }
        
        self._transition_indicators = {
            "gradual": ["also", "additionally", "furthermore", "moreover", "related to"],
            "abrupt": ["but", "however", "instead", "on the other hand", "switching to"],
            "return": ["back to", "returning to", "as we discussed", "earlier topic"],
            "branch": ["regarding", "about", "concerning", "speaking of", "what about"]
        }
    
    def detect_enhanced_topic_shift(
        self, 
        current_query: str, 
        context: ConversationContext
    ) -> EnhancedTopicShift:
        """
        Enhanced topic shift detection with retrieval adaptation recommendations.
        Implements Requirements 6.3 for comprehensive topic change detection.
        """
        
        if not context.recent_messages:
            return EnhancedTopicShift(
                shift_type=TopicShiftType.NO_SHIFT,
                confidence=1.0,
                previous_topics=[],
                current_topics=self._extract_topics_from_text(current_query),
                topic_similarity_score=0.0,
                retrieval_adaptation_needed=False,
                suggested_retrieval_strategy="standard",
                context_preservation_level=0.0
            )
        
        # Extract topics from conversation history and current query
        previous_topics = self._extract_conversation_topics(context.recent_messages[-6:])
        current_topics = self._extract_topics_from_text(current_query)
        
        # Calculate topic similarity
        similarity_score = self._calculate_topic_similarity(previous_topics, current_topics)
        
        # Detect shift type
        shift_type = self._classify_shift_type(
            current_query, previous_topics, current_topics, similarity_score
        )
        
        # Calculate confidence
        confidence = self._calculate_shift_confidence(
            shift_type, similarity_score, current_query, context
        )
        
        # Determine if retrieval adaptation is needed
        adaptation_needed = self._needs_retrieval_adaptation(shift_type, similarity_score)
        
        # Suggest retrieval strategy
        retrieval_strategy = self._suggest_retrieval_strategy(shift_type, similarity_score)
        
        # Calculate context preservation level
        preservation_level = self._calculate_context_preservation_level(
            shift_type, similarity_score
        )
        
        return EnhancedTopicShift(
            shift_type=shift_type,
            confidence=confidence,
            previous_topics=previous_topics,
            current_topics=current_topics,
            topic_similarity_score=similarity_score,
            retrieval_adaptation_needed=adaptation_needed,
            suggested_retrieval_strategy=retrieval_strategy,
            context_preservation_level=preservation_level
        )
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text using hierarchical topic detection"""
        
        text_lower = text.lower()
        detected_topics = []
        
        for main_topic, subtopics in self._topic_hierarchies.items():
            main_topic_score = 0
            
            for subtopic, keywords in subtopics.items():
                keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
                
                if keyword_matches > 0:
                    detected_topics.append(subtopic)
                    main_topic_score += keyword_matches
            
            # Add main topic if enough subtopic matches
            if main_topic_score >= 2:
                detected_topics.append(main_topic)
        
        return list(set(detected_topics))  # Remove duplicates
    
    def _extract_conversation_topics(self, messages: List[BaseMessage]) -> List[str]:
        """Extract topics from conversation messages"""
        
        all_topics = []
        
        for message in messages:
            if hasattr(message, 'content'):
                topics = self._extract_topics_from_text(message.content)
                all_topics.extend(topics)
        
        # Return unique topics, preserving order
        seen = set()
        unique_topics = []
        for topic in all_topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)
        
        return unique_topics
    
    def _calculate_topic_similarity(self, previous_topics: List[str], current_topics: List[str]) -> float:
        """Calculate similarity between topic sets"""
        
        if not previous_topics and not current_topics:
            return 1.0
        
        if not previous_topics or not current_topics:
            return 0.0
        
        # Direct overlap
        overlap = len(set(previous_topics) & set(current_topics))
        total_unique = len(set(previous_topics) | set(current_topics))
        
        if total_unique == 0:
            return 1.0
        
        direct_similarity = overlap / total_unique
        
        # Hierarchical similarity (topics in same category)
        hierarchical_similarity = self._calculate_hierarchical_similarity(
            previous_topics, current_topics
        )
        
        # Combine similarities
        return (direct_similarity * 0.7) + (hierarchical_similarity * 0.3)
    
    def _calculate_hierarchical_similarity(self, topics1: List[str], topics2: List[str]) -> float:
        """Calculate similarity based on topic hierarchy"""
        
        related_pairs = 0
        total_pairs = 0
        
        for topic1 in topics1:
            for topic2 in topics2:
                total_pairs += 1
                
                # Check if topics are in the same hierarchy
                if self._are_topics_hierarchically_related(topic1, topic2):
                    related_pairs += 1
        
        if total_pairs == 0:
            return 0.0
        
        return related_pairs / total_pairs
    
    def _are_topics_hierarchically_related(self, topic1: str, topic2: str) -> bool:
        """Check if two topics are hierarchically related"""
        
        for main_topic, subtopics in self._topic_hierarchies.items():
            # Check if both topics are subtopics of the same main topic
            topic1_in_main = topic1 == main_topic or topic1 in subtopics
            topic2_in_main = topic2 == main_topic or topic2 in subtopics
            
            if topic1_in_main and topic2_in_main:
                return True
        
        return False
    
    def _classify_shift_type(
        self, 
        current_query: str, 
        previous_topics: List[str], 
        current_topics: List[str], 
        similarity_score: float
    ) -> TopicShiftType:
        """Classify the type of topic shift"""
        
        query_lower = current_query.lower()
        
        # Check for explicit transition indicators
        for transition_type, indicators in self._transition_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                if transition_type == "gradual":
                    return TopicShiftType.GRADUAL_EVOLUTION
                elif transition_type == "abrupt":
                    return TopicShiftType.ABRUPT_CHANGE
                elif transition_type == "return":
                    return TopicShiftType.RETURN_TO_PREVIOUS
                elif transition_type == "branch":
                    return TopicShiftType.RELATED_BRANCH
        
        # Check for multi-topic queries
        if len(current_topics) > 2 and len(previous_topics) > 0:
            return TopicShiftType.MULTI_TOPIC_QUERY
        
        # Classify based on similarity score
        if similarity_score >= 0.8:
            return TopicShiftType.NO_SHIFT
        elif similarity_score >= 0.5:
            return TopicShiftType.GRADUAL_EVOLUTION
        elif similarity_score >= 0.2:
            return TopicShiftType.RELATED_BRANCH
        else:
            return TopicShiftType.ABRUPT_CHANGE
    
    def _calculate_shift_confidence(
        self, 
        shift_type: TopicShiftType, 
        similarity_score: float, 
        current_query: str,
        context: ConversationContext
    ) -> float:
        """Calculate confidence in shift detection"""
        
        base_confidence = 0.7
        
        # Adjust based on shift type clarity
        if shift_type == TopicShiftType.NO_SHIFT and similarity_score > 0.9:
            base_confidence += 0.2
        elif shift_type == TopicShiftType.ABRUPT_CHANGE and similarity_score < 0.1:
            base_confidence += 0.2
        
        # Adjust based on explicit indicators
        query_lower = current_query.lower()
        has_explicit_indicators = any(
            any(indicator in query_lower for indicator in indicators)
            for indicators in self._transition_indicators.values()
        )
        
        if has_explicit_indicators:
            base_confidence += 0.1
        
        # Adjust based on conversation length
        if len(context.recent_messages) >= 6:
            base_confidence += 0.1  # More context = higher confidence
        
        return min(base_confidence, 1.0)
    
    def _needs_retrieval_adaptation(self, shift_type: TopicShiftType, similarity_score: float) -> bool:
        """Determine if retrieval adaptation is needed"""
        
        adaptation_needed_types = [
            TopicShiftType.ABRUPT_CHANGE,
            TopicShiftType.RELATED_BRANCH,
            TopicShiftType.MULTI_TOPIC_QUERY
        ]
        
        return (shift_type in adaptation_needed_types or 
                similarity_score < 0.3)
    
    def _suggest_retrieval_strategy(self, shift_type: TopicShiftType, similarity_score: float) -> str:
        """Suggest retrieval strategy based on shift type"""
        
        strategy_mapping = {
            TopicShiftType.NO_SHIFT: "standard",
            TopicShiftType.GRADUAL_EVOLUTION: "expand_context",
            TopicShiftType.RELATED_BRANCH: "broaden_scope",
            TopicShiftType.ABRUPT_CHANGE: "fresh_search",
            TopicShiftType.RETURN_TO_PREVIOUS: "include_historical_context",
            TopicShiftType.MULTI_TOPIC_QUERY: "multi_faceted_search"
        }
        
        return strategy_mapping.get(shift_type, "standard")
    
    def _calculate_context_preservation_level(
        self, 
        shift_type: TopicShiftType, 
        similarity_score: float
    ) -> float:
        """Calculate how much previous context should be preserved"""
        
        preservation_levels = {
            TopicShiftType.NO_SHIFT: 1.0,
            TopicShiftType.GRADUAL_EVOLUTION: 0.8,
            TopicShiftType.RELATED_BRANCH: 0.6,
            TopicShiftType.ABRUPT_CHANGE: 0.2,
            TopicShiftType.RETURN_TO_PREVIOUS: 0.9,
            TopicShiftType.MULTI_TOPIC_QUERY: 0.7
        }
        
        base_level = preservation_levels.get(shift_type, 0.5)
        
        # Adjust based on similarity score
        adjusted_level = (base_level + similarity_score) / 2
        
        return min(adjusted_level, 1.0)


class RetrievalAdaptationEngine:
    """
    Adapts retrieval strategy based on topic shifts and clarification requests.
    Implements Requirements 6.3 for retrieval adaptation to topic changes.
    """
    
    def __init__(self, base_config: RetrievalConfig):
        self.base_config = base_config
    
    def adapt_retrieval_for_topic_shift(
        self, 
        topic_shift: EnhancedTopicShift,
        base_config: RetrievalConfig
    ) -> RetrievalAdaptation:
        """
        Adapt retrieval strategy based on detected topic shift.
        Implements Requirements 6.3 for retrieval adaptation to topic changes.
        """
        
        adaptation = RetrievalAdaptation(
            expand_search_scope=False,
            include_previous_context=True,
            adjust_similarity_threshold=0.0,
            modify_top_k=0,
            add_topic_keywords=[],
            prioritize_recent_docs=False
        )
        
        # Adapt based on shift type
        if topic_shift.shift_type == TopicShiftType.ABRUPT_CHANGE:
            adaptation.expand_search_scope = True
            adaptation.include_previous_context = False
            adaptation.adjust_similarity_threshold = -0.1  # Lower threshold for broader search
            adaptation.modify_top_k = 3  # Increase top_k
            adaptation.add_topic_keywords = topic_shift.current_topics
            
        elif topic_shift.shift_type == TopicShiftType.RELATED_BRANCH:
            adaptation.expand_search_scope = True
            adaptation.include_previous_context = True
            adaptation.adjust_similarity_threshold = -0.05
            adaptation.modify_top_k = 2
            adaptation.add_topic_keywords = topic_shift.current_topics
            
        elif topic_shift.shift_type == TopicShiftType.MULTI_TOPIC_QUERY:
            adaptation.expand_search_scope = True
            adaptation.include_previous_context = True
            adaptation.adjust_similarity_threshold = -0.05
            adaptation.modify_top_k = 4  # Significantly increase for multi-topic
            adaptation.add_topic_keywords = topic_shift.current_topics
            adaptation.prioritize_recent_docs = False
            
        elif topic_shift.shift_type == TopicShiftType.RETURN_TO_PREVIOUS:
            adaptation.expand_search_scope = False
            adaptation.include_previous_context = True
            adaptation.adjust_similarity_threshold = 0.05  # Higher threshold for precision
            adaptation.modify_top_k = -1  # Slightly reduce top_k
            adaptation.prioritize_recent_docs = False
            
        elif topic_shift.shift_type == TopicShiftType.GRADUAL_EVOLUTION:
            adaptation.expand_search_scope = False
            adaptation.include_previous_context = True
            adaptation.adjust_similarity_threshold = 0.0
            adaptation.modify_top_k = 1
            adaptation.add_topic_keywords = topic_shift.current_topics[:2]  # Limit keywords
        
        # Adjust based on context preservation level
        if topic_shift.context_preservation_level < 0.3:
            adaptation.include_previous_context = False
        elif topic_shift.context_preservation_level > 0.8:
            adaptation.prioritize_recent_docs = True
        
        return adaptation
    
    def adapt_retrieval_for_clarification(
        self, 
        clarification: ClarificationRequest,
        base_config: RetrievalConfig
    ) -> RetrievalAdaptation:
        """
        Adapt retrieval strategy for clarification requests.
        Implements Requirements 6.4 for clarification-specific retrieval adaptation.
        """
        
        adaptation = RetrievalAdaptation(
            expand_search_scope=False,
            include_previous_context=True,
            adjust_similarity_threshold=0.1,  # Higher precision for clarifications
            modify_top_k=-2,  # Fewer, more precise results
            add_topic_keywords=[],
            prioritize_recent_docs=True
        )
        
        # Adapt based on clarification type
        if clarification.clarification_type == ClarificationType.DEFINITION_REQUEST:
            adaptation.adjust_similarity_threshold = 0.15
            adaptation.modify_top_k = -1
            if clarification.target_concept:
                adaptation.add_topic_keywords = [clarification.target_concept]
                
        elif clarification.clarification_type == ClarificationType.ELABORATION_REQUEST:
            adaptation.expand_search_scope = True
            adaptation.adjust_similarity_threshold = 0.05
            adaptation.modify_top_k = 2
            if clarification.target_concept:
                adaptation.add_topic_keywords = [clarification.target_concept]
                
        elif clarification.clarification_type == ClarificationType.EXAMPLE_REQUEST:
            adaptation.expand_search_scope = True
            adaptation.adjust_similarity_threshold = 0.0
            adaptation.modify_top_k = 3
            adaptation.add_topic_keywords = ["example", "case study", "demonstration"]
            
        elif clarification.clarification_type == ClarificationType.PROCESS_CLARIFICATION:
            adaptation.expand_search_scope = True
            adaptation.adjust_similarity_threshold = 0.0
            adaptation.modify_top_k = 2
            adaptation.add_topic_keywords = ["steps", "process", "procedure", "how to"]
            
        elif clarification.clarification_type == ClarificationType.CONTEXT_CLARIFICATION:
            adaptation.expand_search_scope = False
            adaptation.include_previous_context = True
            adaptation.adjust_similarity_threshold = 0.2
            adaptation.modify_top_k = -3  # Very focused search
            adaptation.prioritize_recent_docs = True
        
        return adaptation
    
    def apply_retrieval_adaptation(
        self, 
        adaptation: RetrievalAdaptation,
        base_config: RetrievalConfig
    ) -> RetrievalConfig:
        """Apply adaptation to create modified retrieval configuration"""
        
        adapted_config = RetrievalConfig(
            base_top_k=base_config.base_top_k,
            max_top_k=base_config.max_top_k,
            min_similarity_threshold=base_config.min_similarity_threshold,
            diversity_threshold=base_config.diversity_threshold,
            rerank_enabled=base_config.rerank_enabled,
            hybrid_search_weight=base_config.hybrid_search_weight
        )
        
        # Apply top_k modification
        new_top_k = base_config.base_top_k + adaptation.modify_top_k
        adapted_config.base_top_k = max(1, min(new_top_k, base_config.max_top_k))
        
        # Apply similarity threshold adjustment
        new_threshold = base_config.min_similarity_threshold + adaptation.adjust_similarity_threshold
        adapted_config.min_similarity_threshold = max(0.1, min(new_threshold, 0.95))
        
        # Apply diversity adjustment for expanded scope
        if adaptation.expand_search_scope:
            adapted_config.diversity_threshold = max(0.8, adapted_config.diversity_threshold)
        
        return adapted_config


# Integration functions
def detect_clarification_request(query: str, context: ConversationContext) -> ClarificationRequest:
    """
    Main function to detect clarification requests.
    Implements Requirements 6.4 for clarification request detection.
    """
    detector = AdvancedClarificationDetector()
    return detector.detect_clarification_request(query, context)


def detect_topic_shift_with_adaptation(
    query: str, 
    context: ConversationContext
) -> EnhancedTopicShift:
    """
    Main function to detect topic shifts with retrieval adaptation.
    Implements Requirements 6.3 for topic change detection and retrieval adaptation.
    """
    detector = EnhancedTopicShiftDetector()
    return detector.detect_enhanced_topic_shift(query, context)


def adapt_retrieval_strategy(
    topic_shift: Optional[EnhancedTopicShift],
    clarification: Optional[ClarificationRequest],
    base_config: RetrievalConfig
) -> RetrievalConfig:
    """
    Main function to adapt retrieval strategy based on topic shifts and clarifications.
    """
    engine = RetrievalAdaptationEngine(base_config)
    
    # Start with base configuration
    adapted_config = base_config
    
    # Apply topic shift adaptation
    if topic_shift and topic_shift.retrieval_adaptation_needed:
        topic_adaptation = engine.adapt_retrieval_for_topic_shift(topic_shift, adapted_config)
        adapted_config = engine.apply_retrieval_adaptation(topic_adaptation, adapted_config)
    
    # Apply clarification adaptation
    if clarification and clarification.is_clarification:
        clarif_adaptation = engine.adapt_retrieval_for_clarification(clarification, adapted_config)
        adapted_config = engine.apply_retrieval_adaptation(clarif_adaptation, adapted_config)
    
    return adapted_config