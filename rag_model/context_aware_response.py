"""
Context-Aware Response Generation for RAG Chatbot
Implements enhanced response generation with context integration and conflict detection.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessage

from rag_model.context_manager import ConversationContext, ConversationComplexity
from rag_model.dynamic_retriever import QueryAnalysis
from rag_model.adaptive_complexity import (
    AdvancedExpertiseDetector, ResponseComplexityAdjuster, 
    ExpertiseProfile, ComplexityAdjustment
)
from rag_model.topic_clarification_handler import (
    AdvancedClarificationDetector, EnhancedTopicShiftDetector,
    ClarificationRequest, EnhancedTopicShift, RetrievalAdaptationEngine
)


class ConflictType(Enum):
    """Types of conflicts between context and retrieved documents"""
    FACTUAL_CONTRADICTION = "factual_contradiction"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    SCOPE_MISMATCH = "scope_mismatch"
    PERSPECTIVE_DIFFERENCE = "perspective_difference"
    NO_CONFLICT = "no_conflict"


class ResponseComplexity(Enum):
    """Levels of response complexity based on user expertise"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class ConflictDetection:
    """Results of conflict detection between context and documents"""
    has_conflict: bool
    conflict_type: ConflictType
    conflicting_elements: List[str]
    confidence_score: float
    resolution_strategy: str


@dataclass
class UserExpertise:
    """User expertise level detection results"""
    level: ResponseComplexity
    confidence: float
    indicators: List[str]
    domain_knowledge: Dict[str, float]


@dataclass
class TopicShift:
    """Topic shift detection results"""
    shift_detected: bool
    previous_topics: List[str]
    current_topics: List[str]
    shift_type: str  # "gradual", "abrupt", "related", "unrelated"
    confidence: float


class ConflictDetector:
    """
    Detects conflicts between conversation context and retrieved documents.
    Implements Requirements 6.2 for conflict detection and explanation.
    """
    
    def __init__(self):
        self._factual_indicators = [
            "is", "are", "was", "were", "has", "have", "will", "would",
            "can", "cannot", "must", "should", "always", "never"
        ]
        
        self._temporal_indicators = [
            "before", "after", "during", "since", "until", "when", "while",
            "now", "today", "yesterday", "tomorrow", "recently", "currently"
        ]
        
        self._contradiction_patterns = [
            r"\b(not|no|never|none|nothing)\b",
            r"\b(but|however|although|despite|contrary)\b",
            r"\b(instead|rather|alternatively)\b",
            r"\b(different|opposite|unlike|versus)\b"
        ]
    
    def detect_conflicts(
        self, 
        context: ConversationContext, 
        retrieved_docs: List[Document],
        current_query: str
    ) -> ConflictDetection:
        """
        Detect conflicts between conversation context and retrieved documents.
        Implements Requirements 6.2 for conflict detection between context and retrieved information.
        """
        
        if not context.recent_messages or not retrieved_docs:
            return ConflictDetection(
                has_conflict=False,
                conflict_type=ConflictType.NO_CONFLICT,
                conflicting_elements=[],
                confidence_score=0.0,
                resolution_strategy="no_conflict"
            )
        
        # Extract key statements from context
        context_statements = self._extract_key_statements(context.recent_messages)
        
        # Extract key statements from retrieved documents
        doc_statements = self._extract_doc_statements(retrieved_docs)
        
        # Compare statements for conflicts
        conflicts = self._compare_statements(context_statements, doc_statements)
        
        if not conflicts:
            return ConflictDetection(
                has_conflict=False,
                conflict_type=ConflictType.NO_CONFLICT,
                conflicting_elements=[],
                confidence_score=0.0,
                resolution_strategy="no_conflict"
            )
        
        # Determine primary conflict type and confidence
        conflict_type, confidence = self._classify_conflict_type(conflicts)
        
        # Determine resolution strategy
        resolution_strategy = self._determine_resolution_strategy(conflict_type, confidence)
        
        return ConflictDetection(
            has_conflict=True,
            conflict_type=conflict_type,
            conflicting_elements=[conflict["description"] for conflict in conflicts],
            confidence_score=confidence,
            resolution_strategy=resolution_strategy
        )
    
    def _extract_key_statements(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Extract key factual statements from conversation messages"""
        statements = []
        
        for message in messages[-6:]:  # Last 3 exchanges
            if isinstance(message, (HumanMessage, AIMessage)):
                content = message.content
                
                # Extract sentences with factual indicators
                sentences = re.split(r'[.!?]+', content)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 10:  # Skip very short sentences
                        continue
                    
                    # Check for factual indicators
                    if any(indicator in sentence.lower() for indicator in self._factual_indicators):
                        statements.append({
                            "text": sentence,
                            "source": "human" if isinstance(message, HumanMessage) else "ai",
                            "type": "factual"
                        })
                    
                    # Check for temporal statements
                    if any(indicator in sentence.lower() for indicator in self._temporal_indicators):
                        statements.append({
                            "text": sentence,
                            "source": "human" if isinstance(message, HumanMessage) else "ai",
                            "type": "temporal"
                        })
        
        return statements
    
    def _extract_doc_statements(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract key statements from retrieved documents"""
        statements = []
        
        for doc in documents:
            content = doc.page_content
            
            # Extract sentences with factual content
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                # Check for factual indicators
                if any(indicator in sentence.lower() for indicator in self._factual_indicators):
                    statements.append({
                        "text": sentence,
                        "source": doc.metadata.get("source", "document"),
                        "type": "factual"
                    })
                
                # Check for temporal statements
                if any(indicator in sentence.lower() for indicator in self._temporal_indicators):
                    statements.append({
                        "text": sentence,
                        "source": doc.metadata.get("source", "document"),
                        "type": "temporal"
                    })
        
        return statements[:20]  # Limit to prevent overwhelming analysis
    
    def _compare_statements(
        self, 
        context_statements: List[Dict[str, Any]], 
        doc_statements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compare statements to identify potential conflicts"""
        conflicts = []
        
        for ctx_stmt in context_statements:
            for doc_stmt in doc_statements:
                # Skip if same type and might be compatible
                if ctx_stmt["type"] != doc_stmt["type"]:
                    continue
                
                conflict = self._analyze_statement_pair(ctx_stmt, doc_stmt)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _analyze_statement_pair(
        self, 
        ctx_stmt: Dict[str, Any], 
        doc_stmt: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze a pair of statements for conflicts"""
        
        ctx_text = ctx_stmt["text"].lower()
        doc_text = doc_stmt["text"].lower()
        
        # Look for direct contradictions
        contradiction_score = 0
        
        # Check for negation patterns
        ctx_has_negation = any(re.search(pattern, ctx_text) for pattern in self._contradiction_patterns)
        doc_has_negation = any(re.search(pattern, doc_text) for pattern in self._contradiction_patterns)
        
        if ctx_has_negation != doc_has_negation:
            contradiction_score += 0.3
        
        # Check for opposite terms
        opposite_pairs = [
            ("yes", "no"), ("true", "false"), ("correct", "incorrect"),
            ("right", "wrong"), ("good", "bad"), ("positive", "negative"),
            ("increase", "decrease"), ("more", "less"), ("higher", "lower")
        ]
        
        for pos, neg in opposite_pairs:
            if ((pos in ctx_text and neg in doc_text) or 
                (neg in ctx_text and pos in doc_text)):
                contradiction_score += 0.4
        
        # Check for temporal conflicts
        if ctx_stmt["type"] == "temporal" and doc_stmt["type"] == "temporal":
            temporal_conflict = self._check_temporal_conflict(ctx_text, doc_text)
            if temporal_conflict:
                contradiction_score += 0.5
        
        # Check for factual conflicts using keyword overlap
        if ctx_stmt["type"] == "factual" and doc_stmt["type"] == "factual":
            factual_conflict = self._check_factual_conflict(ctx_text, doc_text)
            if factual_conflict:
                contradiction_score += factual_conflict
        
        if contradiction_score >= 0.3:  # Threshold for conflict detection
            return {
                "context_statement": ctx_stmt["text"],
                "document_statement": doc_stmt["text"],
                "conflict_score": contradiction_score,
                "description": f"Potential conflict between context and document statements"
            }
        
        return None
    
    def _check_temporal_conflict(self, ctx_text: str, doc_text: str) -> bool:
        """Check for temporal conflicts between statements"""
        
        # Simple temporal conflict detection
        temporal_opposites = [
            ("before", "after"), ("past", "future"), ("old", "new"),
            ("previous", "next"), ("earlier", "later")
        ]
        
        for early, late in temporal_opposites:
            if ((early in ctx_text and late in doc_text) or 
                (late in ctx_text and early in doc_text)):
                return True
        
        return False
    
    def _check_factual_conflict(self, ctx_text: str, doc_text: str) -> float:
        """Check for factual conflicts and return conflict strength"""
        
        # Extract key terms from both statements
        ctx_words = set(re.findall(r'\b\w+\b', ctx_text.lower()))
        doc_words = set(re.findall(r'\b\w+\b', doc_text.lower()))
        
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
            "for", "of", "with", "by", "is", "are", "was", "were"
        }
        ctx_words -= stop_words
        doc_words -= stop_words
        
        # Calculate overlap
        overlap = len(ctx_words & doc_words)
        total_unique = len(ctx_words | doc_words)
        
        if total_unique == 0:
            return 0.0
        
        overlap_ratio = overlap / total_unique
        
        # High overlap with contradictory indicators suggests conflict
        if overlap_ratio > 0.3:
            # Check for contradictory patterns in the overlapping context
            combined_text = f"{ctx_text} {doc_text}"
            contradiction_indicators = sum(
                1 for pattern in self._contradiction_patterns 
                if re.search(pattern, combined_text)
            )
            
            return min(overlap_ratio * contradiction_indicators * 0.2, 0.8)
        
        return 0.0
    
    def _classify_conflict_type(self, conflicts: List[Dict[str, Any]]) -> Tuple[ConflictType, float]:
        """Classify the primary type of conflict and calculate confidence"""
        
        if not conflicts:
            return ConflictType.NO_CONFLICT, 0.0
        
        # Calculate average conflict score
        avg_score = sum(conflict["conflict_score"] for conflict in conflicts) / len(conflicts)
        
        # Simple classification based on content analysis
        conflict_texts = " ".join([
            conflict["context_statement"] + " " + conflict["document_statement"] 
            for conflict in conflicts
        ]).lower()
        
        # Check for temporal indicators
        if any(indicator in conflict_texts for indicator in self._temporal_indicators):
            return ConflictType.TEMPORAL_INCONSISTENCY, avg_score
        
        # Check for factual contradictions
        if any(re.search(pattern, conflict_texts) for pattern in self._contradiction_patterns):
            return ConflictType.FACTUAL_CONTRADICTION, avg_score
        
        # Default to factual contradiction
        return ConflictType.FACTUAL_CONTRADICTION, avg_score
    
    def _determine_resolution_strategy(self, conflict_type: ConflictType, confidence: float) -> str:
        """Determine the best strategy for resolving the conflict"""
        
        if confidence < 0.4:
            return "acknowledge_uncertainty"
        elif confidence < 0.7:
            return "present_both_perspectives"
        else:
            if conflict_type == ConflictType.TEMPORAL_INCONSISTENCY:
                return "clarify_timeline"
            elif conflict_type == ConflictType.FACTUAL_CONTRADICTION:
                return "prioritize_authoritative_source"
            else:
                return "explain_different_contexts"


class UserExpertiseDetector:
    """
    Detects user expertise level from conversation context.
    Implements Requirements 6.5 for user expertise detection and response complexity adjustment.
    """
    
    def __init__(self):
        self._beginner_indicators = [
            "what is", "how do i", "explain", "simple", "basic", "beginner",
            "new to", "don't understand", "confused", "help me understand"
        ]
        
        self._intermediate_indicators = [
            "how does", "why does", "difference between", "compare", "best practice",
            "recommend", "should i", "which is better"
        ]
        
        self._advanced_indicators = [
            "optimize", "performance", "architecture", "design pattern", "algorithm",
            "implementation", "scalability", "efficiency", "trade-off"
        ]
        
        self._expert_indicators = [
            "benchmark", "profiling", "low-level", "internals", "protocol",
            "specification", "edge case", "corner case", "optimization"
        ]
        
        self._technical_domains = {
            "programming": ["code", "function", "variable", "class", "method", "api"],
            "data": ["database", "query", "table", "schema", "index", "sql"],
            "system": ["server", "network", "security", "deployment", "infrastructure"],
            "ai_ml": ["model", "training", "algorithm", "neural", "machine learning"]
        }
    
    def detect_expertise_level(self, context: ConversationContext) -> UserExpertise:
        """
        Detect user expertise level from conversation context.
        Implements Requirements 6.5 for user expertise detection from context.
        """
        
        if not context.recent_messages:
            return UserExpertise(
                level=ResponseComplexity.INTERMEDIATE,
                confidence=0.5,
                indicators=[],
                domain_knowledge={}
            )
        
        # Extract user messages for analysis
        user_messages = [
            msg.content for msg in context.recent_messages 
            if isinstance(msg, HumanMessage)
        ]
        
        if not user_messages:
            return UserExpertise(
                level=ResponseComplexity.INTERMEDIATE,
                confidence=0.5,
                indicators=[],
                domain_knowledge={}
            )
        
        # Combine all user messages for analysis
        combined_text = " ".join(user_messages).lower()
        
        # Count indicators for each level
        level_scores = {
            ResponseComplexity.BEGINNER: self._count_indicators(combined_text, self._beginner_indicators),
            ResponseComplexity.INTERMEDIATE: self._count_indicators(combined_text, self._intermediate_indicators),
            ResponseComplexity.ADVANCED: self._count_indicators(combined_text, self._advanced_indicators),
            ResponseComplexity.EXPERT: self._count_indicators(combined_text, self._expert_indicators)
        }
        
        # Analyze message characteristics
        avg_message_length = sum(len(msg.split()) for msg in user_messages) / len(user_messages)
        technical_term_count = sum(
            sum(1 for term in terms if term in combined_text)
            for terms in self._technical_domains.values()
        )
        
        # Adjust scores based on message characteristics
        if avg_message_length > 20:
            level_scores[ResponseComplexity.ADVANCED] += 1
            level_scores[ResponseComplexity.EXPERT] += 1
        elif avg_message_length < 8:
            level_scores[ResponseComplexity.BEGINNER] += 1
        
        if technical_term_count > 5:
            level_scores[ResponseComplexity.ADVANCED] += 2
            level_scores[ResponseComplexity.EXPERT] += 1
        elif technical_term_count < 2:
            level_scores[ResponseComplexity.BEGINNER] += 1
        
        # Determine primary level
        max_level = max(level_scores.items(), key=lambda x: x[1])
        detected_level = max_level[0]
        
        # Calculate confidence based on score distribution
        total_score = sum(level_scores.values())
        confidence = max_level[1] / max(total_score, 1) if total_score > 0 else 0.5
        
        # Extract indicators that led to this classification
        indicators = self._extract_matching_indicators(combined_text, detected_level)
        
        # Analyze domain knowledge
        domain_knowledge = self._analyze_domain_knowledge(combined_text)
        
        return UserExpertise(
            level=detected_level,
            confidence=min(confidence * 1.5, 1.0),  # Boost confidence slightly
            indicators=indicators,
            domain_knowledge=domain_knowledge
        )
    
    def _count_indicators(self, text: str, indicators: List[str]) -> int:
        """Count how many indicators are present in the text"""
        return sum(1 for indicator in indicators if indicator in text)
    
    def _extract_matching_indicators(self, text: str, level: ResponseComplexity) -> List[str]:
        """Extract the specific indicators that matched for the detected level"""
        
        indicator_lists = {
            ResponseComplexity.BEGINNER: self._beginner_indicators,
            ResponseComplexity.INTERMEDIATE: self._intermediate_indicators,
            ResponseComplexity.ADVANCED: self._advanced_indicators,
            ResponseComplexity.EXPERT: self._expert_indicators
        }
        
        indicators = indicator_lists.get(level, [])
        return [indicator for indicator in indicators if indicator in text]
    
    def _analyze_domain_knowledge(self, text: str) -> Dict[str, float]:
        """Analyze domain-specific knowledge level"""
        
        domain_scores = {}
        
        for domain, terms in self._technical_domains.items():
            term_count = sum(1 for term in terms if term in text)
            # Normalize by domain size and text length
            score = min(term_count / len(terms), 1.0)
            domain_scores[domain] = score
        
        return domain_scores


class TopicShiftDetector:
    """
    Detects topic shifts in conversation and adapts retrieval strategy.
    Implements Requirements 6.3 for topic change detection and retrieval adaptation.
    """
    
    def __init__(self):
        self._topic_keywords = {
            "technical": ["api", "code", "function", "error", "debug", "implementation"],
            "business": ["strategy", "market", "revenue", "customer", "product", "sales"],
            "support": ["help", "issue", "problem", "fix", "troubleshoot", "resolve"],
            "general": ["what", "how", "why", "when", "where", "explain", "describe"],
            "data": ["database", "query", "table", "analysis", "report", "metrics"],
            "system": ["server", "network", "security", "deployment", "infrastructure"]
        }
    
    def detect_topic_shift(
        self, 
        current_query: str, 
        context: ConversationContext
    ) -> TopicShift:
        """
        Detect topic shifts between current query and conversation context.
        Implements Requirements 6.3 for topic change detection.
        """
        
        if not context.recent_messages:
            return TopicShift(
                shift_detected=False,
                previous_topics=[],
                current_topics=self._extract_topics(current_query),
                shift_type="initial",
                confidence=1.0
            )
        
        # Extract topics from recent conversation
        previous_topics = self._extract_conversation_topics(context.recent_messages[-6:])
        
        # Extract topics from current query
        current_topics = self._extract_topics(current_query)
        
        # Calculate topic overlap
        previous_set = set(previous_topics)
        current_set = set(current_topics)
        
        overlap = len(previous_set & current_set)
        total_topics = len(previous_set | current_set)
        
        if total_topics == 0:
            overlap_ratio = 1.0
        else:
            overlap_ratio = overlap / total_topics
        
        # Determine if shift occurred
        shift_detected = overlap_ratio < 0.3  # Less than 30% overlap indicates shift
        
        # Classify shift type
        shift_type = self._classify_shift_type(previous_topics, current_topics, overlap_ratio)
        
        # Calculate confidence
        confidence = self._calculate_shift_confidence(
            previous_topics, current_topics, overlap_ratio, context
        )
        
        return TopicShift(
            shift_detected=shift_detected,
            previous_topics=previous_topics,
            current_topics=current_topics,
            shift_type=shift_type,
            confidence=confidence
        )
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from a single text"""
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in self._topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    def _extract_conversation_topics(self, messages: List[BaseMessage]) -> List[str]:
        """Extract topics from conversation messages"""
        
        all_topics = []
        
        for message in messages:
            if hasattr(message, 'content'):
                topics = self._extract_topics(message.content)
                all_topics.extend(topics)
        
        # Return unique topics, preserving order
        seen = set()
        unique_topics = []
        for topic in all_topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)
        
        return unique_topics
    
    def _classify_shift_type(
        self, 
        previous_topics: List[str], 
        current_topics: List[str], 
        overlap_ratio: float
    ) -> str:
        """Classify the type of topic shift"""
        
        if overlap_ratio >= 0.7:
            return "no_shift"
        elif overlap_ratio >= 0.3:
            return "gradual"
        elif overlap_ratio > 0:
            # Check if topics are related
            if self._are_topics_related(previous_topics, current_topics):
                return "related"
            else:
                return "abrupt"
        else:
            return "unrelated"
    
    def _are_topics_related(self, topics1: List[str], topics2: List[str]) -> bool:
        """Check if two sets of topics are conceptually related"""
        
        # Define related topic groups
        related_groups = [
            {"technical", "system", "data"},
            {"business", "support"},
            {"general"}  # General can relate to anything
        ]
        
        # Check if topics fall within the same related group
        for group in related_groups:
            if (any(topic in group for topic in topics1) and 
                any(topic in group for topic in topics2)):
                return True
        
        return False
    
    def _calculate_shift_confidence(
        self, 
        previous_topics: List[str], 
        current_topics: List[str], 
        overlap_ratio: float,
        context: ConversationContext
    ) -> float:
        """Calculate confidence in topic shift detection"""
        
        base_confidence = 0.7
        
        # Adjust based on overlap ratio
        if overlap_ratio < 0.1:
            base_confidence += 0.2  # Very clear shift
        elif overlap_ratio > 0.5:
            base_confidence -= 0.2  # Unclear shift
        
        # Adjust based on topic clarity
        if len(current_topics) >= 2:
            base_confidence += 0.1  # Clear current topics
        if len(previous_topics) >= 2:
            base_confidence += 0.1  # Clear previous topics
        
        # Adjust based on conversation length
        if len(context.recent_messages) >= 6:
            base_confidence += 0.1  # More context available
        
        return max(0.1, min(1.0, base_confidence))


class ContextAwareResponseGenerator:
    """
    Main class for context-aware response generation with conflict detection and adaptive complexity.
    Implements Requirements 6.1, 6.2, 6.3, 6.4, 6.5 for enhanced response generation.
    """
    
    def __init__(self, model_id: str = "meta.llama3-8b-instruct-v1:0", region_name: str = "ap-south-1"):
        self.model_id = model_id
        self.region_name = region_name
        
        # Initialize components
        self.conflict_detector = ConflictDetector()
        self.expertise_detector = AdvancedExpertiseDetector()
        self.complexity_adjuster = ResponseComplexityAdjuster()
        self.clarification_detector = AdvancedClarificationDetector()
        self.topic_shift_detector = EnhancedTopicShiftDetector()
        self.retrieval_adapter = RetrievalAdaptationEngine(None)  # Will be set per request
        
        # Initialize LLM
        self._llm = ChatBedrock(
            model_id=model_id,
            region_name=region_name,
            model_kwargs={"temperature": 0.1}
        )
        
        # Create enhanced prompt template
        self._create_enhanced_prompt_template()
    
    def _create_enhanced_prompt_template(self):
        """Create enhanced prompt template with context integration"""
        
        self.enhanced_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a knowledgeable and context-aware AI assistant. Your responses should:

1. **Context Integration**: Consider the conversation history and build upon previous exchanges naturally
2. **Conflict Handling**: When information conflicts between context and documents, acknowledge and explain differences
3. **Adaptive Complexity**: Adjust your response complexity based on the user's apparent expertise level
4. **Topic Awareness**: Recognize topic shifts and adapt your retrieval and response strategy accordingly
5. **Clarification Support**: Reference previous responses when users ask for clarifications

**Response Guidelines:**
- Be conversational and natural, building on the conversation flow
- When conflicts exist, present both perspectives and explain the differences
- Adjust technical depth based on user expertise indicators
- For topic shifts, acknowledge the change and provide appropriate context
- For clarifications, reference specific parts of previous responses
- Always prioritize accuracy while being helpful and engaging

**Conflict Resolution Strategies:**
- acknowledge_uncertainty: "There seems to be some uncertainty about this..."
- present_both_perspectives: "I found different perspectives on this..."
- clarify_timeline: "Let me clarify the timeline here..."
- prioritize_authoritative_source: "Based on the most authoritative source..."
- explain_different_contexts: "This might depend on the specific context..."

**Complexity Levels:**
- Beginner: Use simple language, provide definitions, include examples
- Intermediate: Balanced technical depth, explain key concepts
- Advanced: Technical detail, assume domain knowledge, focus on implementation
- Expert: Deep technical discussion, edge cases, optimization considerations
"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate.from_template("""
Context Analysis:
{context_analysis}

Retrieved Information:
{context}

Current Question: {input}

Please provide a response that integrates the conversation context appropriately and addresses any conflicts or topic shifts as needed.
""")
        ])
    
    def generate_enhanced_response(
        self,
        query: str,
        retrieved_docs: List[Document],
        context: ConversationContext,
        query_analysis: Optional[QueryAnalysis] = None
    ) -> Dict[str, Any]:
        """
        Generate context-aware response with conflict detection and adaptive complexity.
        Implements Requirements 6.1, 6.2, 6.3, 6.4, 6.5 for comprehensive context-aware response generation.
        """
        
        # Step 1: Detect conflicts between context and retrieved documents
        conflict_detection = self.conflict_detector.detect_conflicts(context, retrieved_docs, query)
        
        # Step 2: Detect user expertise level and determine complexity adjustments
        user_expertise = self.expertise_detector.analyze_user_expertise(context)
        complexity_adjustment = self.complexity_adjuster.determine_response_complexity(user_expertise)
        
        # Step 3: Detect topic shifts and clarification requests
        topic_shift = self.topic_shift_detector.detect_enhanced_topic_shift(query, context)
        clarification_request = self.clarification_detector.detect_clarification_request(query, context)
        
        # Step 4: Prepare context analysis for the prompt
        context_analysis = self._prepare_context_analysis(
            conflict_detection, user_expertise, complexity_adjustment, 
            topic_shift, clarification_request, query_analysis
        )
        
        # Step 5: Format retrieved documents
        formatted_context = self._format_retrieved_documents(retrieved_docs)
        
        # Step 6: Prepare conversation history
        chat_history = context.recent_messages if context.recent_messages else []
        
        # Step 7: Handle clarification requests if needed
        if clarification_request.is_clarification:
            generated_response = self._enhance_response_for_clarification(
                generated_response, clarification_request, context
            )
        # Step 8: Generate response using enhanced prompt
        try:
            response = self._llm.invoke(
                self.enhanced_prompt.format_messages(
                    context_analysis=context_analysis,
                    context=formatted_context,
                    input=query,
                    chat_history=chat_history
                )
            )
            
            generated_response = response.content
            
        except Exception as e:
            print(f"Enhanced response generation failed: {e}")
            # Fallback to basic response
            generated_response = self._generate_fallback_response(query, retrieved_docs, context)
        
        # Step 9: Post-process response based on detected characteristics
        final_response = self._post_process_response(
            generated_response, user_expertise, complexity_adjustment, 
            conflict_detection, topic_shift, clarification_request
        )
        
        # Step 10: Compile response metadata
        response_metadata = {
            "conflict_detection": {
                "has_conflict": conflict_detection.has_conflict,
                "conflict_type": conflict_detection.conflict_type.value,
                "resolution_strategy": conflict_detection.resolution_strategy,
                "confidence": conflict_detection.confidence_score
            },
            "user_expertise": {
                "overall_level": user_expertise.overall_level.value,
                "confidence": user_expertise.confidence,
                "domain_expertise": {k: v.value for k, v in user_expertise.domain_expertise.items()},
                "communication_style": user_expertise.communication_style.value,
                "preferred_detail_level": user_expertise.preferred_detail_level,
                "technical_vocabulary_comfort": user_expertise.technical_vocabulary_comfort
            },
            "topic_shift": {
                "shift_detected": topic_shift.shift_type.value != "no_shift",
                "shift_type": topic_shift.shift_type.value,
                "confidence": topic_shift.confidence,
                "previous_topics": topic_shift.previous_topics,
                "current_topics": topic_shift.current_topics,
                "retrieval_adaptation_needed": topic_shift.retrieval_adaptation_needed
            },
            "clarification_request": {
                "is_clarification": clarification_request.is_clarification,
                "clarification_type": clarification_request.clarification_type.value if clarification_request.is_clarification else None,
                "target_concept": clarification_request.target_concept,
                "confidence": clarification_request.confidence,
                "response_approach": clarification_request.suggested_response_approach
            },
            "response_characteristics": {
                "complexity_adjusted": user_expertise.confidence > 0.6,
                "conflict_addressed": conflict_detection.has_conflict,
                "topic_shift_handled": topic_shift.shift_type.value != "no_shift",
                "clarification_handled": clarification_request.is_clarification,
                "complexity_level": complexity_adjustment.vocabulary_level,
                "explanation_depth": complexity_adjustment.explanation_depth
            }
        }
        
        return {
            "response": final_response,
            "metadata": response_metadata,
            "sources": [doc.metadata.get("source", "unknown") for doc in retrieved_docs]
        }
    
    def _prepare_context_analysis(
        self,
        conflict_detection: ConflictDetection,
        user_expertise: ExpertiseProfile,
        complexity_adjustment: ComplexityAdjustment,
        topic_shift: EnhancedTopicShift,
        clarification_request: ClarificationRequest,
        query_analysis: Optional[QueryAnalysis]
    ) -> str:
        """Prepare context analysis summary for the prompt"""
        
        analysis_parts = []
        
        # User expertise analysis
        analysis_parts.append(f"User Expertise: {user_expertise.overall_level.value} (confidence: {user_expertise.confidence:.2f})")
        analysis_parts.append(f"Communication Style: {user_expertise.communication_style.value}")
        analysis_parts.append(f"Preferred Detail Level: {user_expertise.preferred_detail_level}")
        
        if user_expertise.domain_expertise:
            domain_info = ", ".join([f"{domain}: {level.value}" for domain, level in user_expertise.domain_expertise.items()])
            analysis_parts.append(f"Domain Expertise: {domain_info}")
        
        # Complexity adjustment guidance
        complexity_guidance = self.complexity_adjuster.create_complexity_prompt_additions(complexity_adjustment)
        analysis_parts.append(f"Response Complexity Guidance:\n{complexity_guidance}")
        
        # Conflict detection
        if conflict_detection.has_conflict:
            analysis_parts.append(f"Conflict Detected: {conflict_detection.conflict_type.value}")
            analysis_parts.append(f"Resolution Strategy: {conflict_detection.resolution_strategy}")
            if conflict_detection.conflicting_elements:
                analysis_parts.append(f"Conflicting Elements: {conflict_detection.conflicting_elements[0][:100]}...")
        else:
            analysis_parts.append("No conflicts detected between context and retrieved information")
        
        # Topic shift analysis
        if topic_shift.shift_type.value != "no_shift":
            analysis_parts.append(f"Topic Shift: {topic_shift.shift_type.value} shift detected")
            analysis_parts.append(f"Previous Topics: {', '.join(topic_shift.previous_topics)}")
            analysis_parts.append(f"Current Topics: {', '.join(topic_shift.current_topics)}")
            if topic_shift.retrieval_adaptation_needed:
                analysis_parts.append(f"Retrieval Strategy: {topic_shift.suggested_retrieval_strategy}")
        else:
            analysis_parts.append("No significant topic shift detected")
        
        # Clarification request analysis
        if clarification_request.is_clarification:
            analysis_parts.append(f"Clarification Request: {clarification_request.clarification_type.value}")
            analysis_parts.append(f"Target Concept: {clarification_request.target_concept}")
            analysis_parts.append(f"Response Approach: {clarification_request.suggested_response_approach}")
            if clarification_request.reference_message_index >= 0:
                analysis_parts.append("References previous response")
        else:
            analysis_parts.append("No clarification request detected")
        
        # Query analysis if available
        if query_analysis:
            analysis_parts.append(f"Query Complexity: {query_analysis.complexity.value}")
            analysis_parts.append(f"Query Type: {query_analysis.query_type.value}")
            if query_analysis.requires_context:
                analysis_parts.append("Query requires conversation context")
        
        return "\n".join(analysis_parts)
    
    def _format_retrieved_documents(self, documents: List[Document]) -> str:
        """Format retrieved documents for the prompt"""
        
        if not documents:
            return "No relevant documents retrieved."
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", f"Document {i}")
            content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            formatted_docs.append(f"Source {i} ({source}):\n{content}")
        
        return "\n\n".join(formatted_docs)
    
    def _generate_fallback_response(
        self, 
        query: str, 
        retrieved_docs: List[Document], 
        context: ConversationContext
    ) -> str:
        """Generate a fallback response when enhanced generation fails"""
        
        if not retrieved_docs:
            return "I don't have enough information to answer your question. Could you provide more context or rephrase your question?"
        
        # Simple response based on retrieved documents
        doc_content = " ".join([doc.page_content[:200] for doc in retrieved_docs[:2]])
        
        return f"Based on the available information: {doc_content[:400]}..."
    
    def _post_process_response(
        self,
        response: str,
        user_expertise: ExpertiseProfile,
        complexity_adjustment: ComplexityAdjustment,
        conflict_detection: ConflictDetection,
        topic_shift: EnhancedTopicShift,
        clarification_request: ClarificationRequest
    ) -> str:
        """Post-process the response based on detected characteristics"""
        
        processed_response = response
        
        # Add expertise-appropriate adjustments based on complexity adjustment
        if complexity_adjustment.include_definitions and user_expertise.confidence > 0.7:
            processed_response = self._ensure_definitions_included(processed_response)
        
        if complexity_adjustment.use_analogies and user_expertise.confidence > 0.7:
            processed_response = self._enhance_with_analogies(processed_response)
        
        if complexity_adjustment.provide_step_by_step and user_expertise.confidence > 0.7:
            processed_response = self._ensure_step_by_step_format(processed_response)
        
        # Add conflict acknowledgment if not already present
        if conflict_detection.has_conflict and conflict_detection.confidence > 0.6:
            if not any(phrase in processed_response.lower() for phrase in ["conflict", "different", "however", "on the other hand"]):
                processed_response = self._add_conflict_acknowledgment(processed_response, conflict_detection)
        
        # Add topic shift acknowledgment if needed
        if topic_shift.shift_type.value != "no_shift" and topic_shift.confidence > 0.7:
            if not any(phrase in processed_response.lower() for phrase in ["shifting to", "moving to", "regarding"]):
                processed_response = self._add_topic_shift_acknowledgment(processed_response, topic_shift)
        
        # Add clarification-specific enhancements
        if clarification_request.is_clarification and clarification_request.confidence > 0.6:
            processed_response = self._enhance_clarification_response(processed_response, clarification_request)
        
        return processed_response
    
    def _ensure_definitions_included(self, response: str) -> str:
        """Ensure definitions are included for technical terms when needed"""
        # This is a placeholder - in a real implementation, this could identify
        # technical terms and add definitions
        return response
    
    def _enhance_with_analogies(self, response: str) -> str:
        """Enhance response with analogies when appropriate"""
        # This is a placeholder - in a real implementation, this could add
        # analogies for complex concepts
        return response
    
    def _ensure_step_by_step_format(self, response: str) -> str:
        """Ensure response is formatted in step-by-step manner when needed"""
        # This is a placeholder - in a real implementation, this could
        # restructure responses into numbered steps
        return response
    
    def _add_beginner_context(self, response: str) -> str:
        """Add beginner-friendly context to the response"""
        
        # Simple addition - in a real implementation, this could be more sophisticated
        if len(response) > 100:
            return f"Let me explain this in simple terms: {response}"
        return response
    
    def _enhance_technical_depth(self, response: str) -> str:
        """Enhance technical depth for expert users"""
        
        # This is a placeholder - real implementation would analyze and enhance technical content
        return response
    
    def _add_conflict_acknowledgment(self, response: str, conflict_detection: ConflictDetection) -> str:
        """Add conflict acknowledgment to the response"""
        
        conflict_intro = {
            "acknowledge_uncertainty": "I notice there's some uncertainty in the available information. ",
            "present_both_perspectives": "I found different perspectives on this topic. ",
            "clarify_timeline": "Let me clarify the timeline to address some inconsistencies. ",
            "prioritize_authoritative_source": "Based on the most authoritative sources available: ",
            "explain_different_contexts": "This might depend on the specific context. "
        }
        
        intro = conflict_intro.get(conflict_detection.resolution_strategy, "I notice some conflicting information. ")
        return intro + response
    
    def _add_topic_shift_acknowledgment(self, response: str, topic_shift: EnhancedTopicShift) -> str:
        """Add topic shift acknowledgment to the response"""
        
        if topic_shift.shift_type.value == "abrupt_change":
            intro = f"I see we're shifting from {', '.join(topic_shift.previous_topics)} to {', '.join(topic_shift.current_topics)}. "
        elif topic_shift.shift_type.value == "related_branch":
            intro = f"Moving to a related topic about {', '.join(topic_shift.current_topics)}: "
        elif topic_shift.shift_type.value == "return_to_previous":
            intro = f"Returning to our earlier discussion about {', '.join(topic_shift.current_topics)}: "
        elif topic_shift.shift_type.value == "multi_topic_query":
            intro = f"I'll address the multiple topics you've raised about {', '.join(topic_shift.current_topics)}: "
        else:
            intro = f"Regarding {', '.join(topic_shift.current_topics)}: "
        
        return intro + response
    
    def _enhance_response_for_clarification(
        self, 
        response: str, 
        clarification_request: ClarificationRequest, 
        context: ConversationContext
    ) -> str:
        """
        Enhance response specifically for clarification requests.
        Implements Requirements 6.4 for reference to previous responses for clarifications.
        """
        
        if not clarification_request.is_clarification:
            return response
        
        # Add reference to previous response if available
        if clarification_request.reference_message_index >= 0:
            try:
                referenced_message = context.recent_messages[clarification_request.reference_message_index]
                if isinstance(referenced_message, AIMessage):
                    reference_text = referenced_message.content[:100] + "..." if len(referenced_message.content) > 100 else referenced_message.content
                    clarification_intro = f"To clarify my previous point about '{reference_text}': "
                    response = clarification_intro + response
            except (IndexError, AttributeError):
                pass  # Fallback to standard response
        
        return response
    
    def _enhance_clarification_response(
        self, 
        response: str, 
        clarification_request: ClarificationRequest
    ) -> str:
        """Add clarification-specific enhancements to the response"""
        
        approach_enhancements = {
            "provide_definition_with_context": "Let me define this clearly: ",
            "expand_with_details": "Let me provide more details: ",
            "provide_concrete_examples": "Here are some specific examples: ",
            "break_down_steps": "Let me break this down step by step: ",
            "structured_comparison": "Let me compare these systematically: ",
            "reference_and_clarify": "To clarify what I mentioned earlier: "
        }
        
        enhancement = approach_enhancements.get(clarification_request.suggested_response_approach, "")
        
        if enhancement and not any(phrase in response.lower() for phrase in ["let me", "to clarify", "here are"]):
            return enhancement + response
        
        return response
    
    def handle_clarification_request(
        self,
        query: str,
        context: ConversationContext,
        previous_response: str
    ) -> str:
        """
        Handle clarification requests by referencing previous responses.
        Implements Requirements 6.4 for reference to previous responses for clarifications.
        """
        
        clarification_patterns = [
            "what do you mean", "can you explain", "clarify", "elaborate",
            "more details", "be more specific", "what exactly", "how so"
        ]
        
        query_lower = query.lower()
        is_clarification = any(pattern in query_lower for pattern in clarification_patterns)
        
        if not is_clarification:
            return ""  # Not a clarification request
        
        # Extract the part of previous response that might need clarification
        response_sentences = re.split(r'[.!?]+', previous_response)
        
        # Simple approach: reference the most relevant sentence
        # In a more sophisticated implementation, this could use semantic similarity
        relevant_sentence = ""
        for sentence in response_sentences:
            if len(sentence.strip()) > 20:  # Skip very short sentences
                relevant_sentence = sentence.strip()
                break
        
        if relevant_sentence:
            return f"To clarify my previous point about '{relevant_sentence[:100]}...': "
        else:
            return "To elaborate on my previous response: "


# Integration function for backward compatibility
def enhance_response_generation(
    query: str,
    retrieved_docs: List[Document],
    context: ConversationContext,
    query_analysis: Optional[QueryAnalysis] = None
) -> Dict[str, Any]:
    """
    Main function to enhance response generation with context integration.
    This function provides a simple interface for the enhanced response generation system.
    """
    
    generator = ContextAwareResponseGenerator()
    return generator.generate_enhanced_response(query, retrieved_docs, context, query_analysis)