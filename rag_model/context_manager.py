"""
Enhanced Context Manager for RAG Chatbot
Implements dynamic context window sizing, conversation complexity analysis,
and adaptive context length adjustment.
"""

import re
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_aws import ChatBedrock
from rag_model.config_models import ContextConfig, QueryComplexity


class ConversationComplexity(Enum):
    """Levels of conversation complexity"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ANALYTICAL = "analytical"


@dataclass
class ContextRequirements:
    """Requirements for conversation context based on query analysis"""
    required_turns: int
    max_tokens: int
    complexity: ConversationComplexity
    needs_summarization: bool
    relevance_threshold: float


@dataclass
class ConversationContext:
    """Enhanced conversation context with metadata"""
    recent_messages: List[BaseMessage]
    conversation_summary: Optional[str]
    topic_context: List[str]
    user_preferences: Dict[str, Any]
    context_tokens: int
    relevance_scores: Dict[str, float]
    complexity: ConversationComplexity


class EnhancedContextManager:
    """
    Enhanced context manager with dynamic window sizing and conversation analysis.
    Implements Requirements 1.1, 1.2, 1.4, 1.5 for conversation memory management.
    """
    
    def __init__(self, config: ContextConfig = None):
        self.config = config or ContextConfig()
        self._topic_keywords = self._load_topic_keywords()
        
        # Initialize LLM for summarization
        self._llm = ChatBedrock(
            model_id="meta.llama3-8b-instruct-v1:0",
            region_name=os.getenv("AWS_DEFAULT_REGION", "ap-south-1"),
            model_kwargs={"temperature": 0.1}
        )
    
    def _load_topic_keywords(self) -> Dict[str, List[str]]:
        """Load topic keywords for context analysis"""
        return {
            "technical": ["api", "code", "function", "error", "debug", "implementation", "algorithm"],
            "business": ["strategy", "market", "revenue", "customer", "product", "sales"],
            "support": ["help", "issue", "problem", "fix", "troubleshoot", "resolve"],
            "general": ["what", "how", "why", "when", "where", "explain", "describe"]
        }
    
    def analyze_conversation_complexity(self, messages: List[BaseMessage]) -> ConversationComplexity:
        """
        Analyze conversation complexity based on message patterns and content.
        Implements Requirements 1.1, 1.2 for dynamic context adjustment.
        """
        if not messages:
            return ConversationComplexity.SIMPLE
        
        # Get recent human messages for analysis
        recent_human_messages = [
            msg.content for msg in messages[-10:] 
            if isinstance(msg, HumanMessage)
        ]
        
        if not recent_human_messages:
            return ConversationComplexity.SIMPLE
        
        complexity_score = 0
        
        # Analyze message characteristics
        for message in recent_human_messages:
            # Length complexity
            if len(message.split()) > 20:
                complexity_score += 2
            elif len(message.split()) > 10:
                complexity_score += 1
            
            # Question complexity
            question_words = len(re.findall(r'\b(what|how|why|when|where|which|who)\b', message.lower()))
            complexity_score += question_words
            
            # Technical terms
            technical_terms = sum(1 for keyword in self._topic_keywords["technical"] 
                                if keyword in message.lower())
            complexity_score += technical_terms * 2
            
            # Multiple topics in one message
            if len(re.findall(r'\band\b|\bor\b|\balso\b|\badditionally\b', message.lower())) > 0:
                complexity_score += 1
            
            # Follow-up indicators
            if any(phrase in message.lower() for phrase in 
                   ["follow up", "also", "additionally", "furthermore", "moreover", "building on"]):
                complexity_score += 2
        
        # Conversation flow complexity
        if len(recent_human_messages) > 5:
            # Check for topic shifts
            topics_per_message = []
            for message in recent_human_messages:
                message_topics = []
                for topic, keywords in self._topic_keywords.items():
                    if any(keyword in message.lower() for keyword in keywords):
                        message_topics.append(topic)
                topics_per_message.append(message_topics)
            
            # Count topic changes
            topic_changes = 0
            for i in range(1, len(topics_per_message)):
                if set(topics_per_message[i]) != set(topics_per_message[i-1]):
                    topic_changes += 1
            
            complexity_score += topic_changes
        
        # Determine complexity level
        if complexity_score >= 15:
            return ConversationComplexity.ANALYTICAL
        elif complexity_score >= 8:
            return ConversationComplexity.COMPLEX
        elif complexity_score >= 3:
            return ConversationComplexity.MODERATE
        else:
            return ConversationComplexity.SIMPLE
    
    def calculate_context_requirements(self, query: str, messages: List[BaseMessage]) -> ContextRequirements:
        """
        Calculate context requirements based on query and conversation complexity.
        Implements Requirements 1.1, 1.2 for adaptive context length adjustment.
        """
        complexity = self.analyze_conversation_complexity(messages)
        
        # Base requirements from configuration
        base_turns = self.config.base_context_turns
        max_tokens = self.config.max_context_tokens
        
        # Adjust based on complexity
        complexity_multipliers = {
            ConversationComplexity.SIMPLE: 0.7,
            ConversationComplexity.MODERATE: 1.0,
            ConversationComplexity.COMPLEX: 1.4,
            ConversationComplexity.ANALYTICAL: 1.8
        }
        
        multiplier = complexity_multipliers[complexity]
        required_turns = min(
            int(base_turns * multiplier),
            self.config.max_context_turns
        )
        
        # Adjust tokens based on query characteristics
        query_length = len(query.split())
        if query_length > 20:
            max_tokens = min(int(max_tokens * 1.2), 6000)
        elif query_length < 5:
            max_tokens = int(max_tokens * 0.8)
        
        # Check if summarization is needed
        needs_summarization = (
            len(messages) > self.config.summarization_threshold or
            complexity in [ConversationComplexity.COMPLEX, ConversationComplexity.ANALYTICAL]
        )
        
        return ContextRequirements(
            required_turns=required_turns,
            max_tokens=max_tokens,
            complexity=complexity,
            needs_summarization=needs_summarization,
            relevance_threshold=self.config.context_relevance_threshold
        )
    
    def estimate_token_count(self, messages: List[BaseMessage]) -> int:
        """
        Estimate token count for messages (rough approximation).
        Uses 1.3 tokens per word as approximation for English text.
        """
        total_words = 0
        for message in messages:
            total_words += len(message.content.split())
        
        # Rough token estimation: 1.3 tokens per word
        return int(total_words * 1.3)
    
    def select_relevant_messages(
        self, 
        messages: List[BaseMessage], 
        requirements: ContextRequirements,
        current_query: str
    ) -> List[BaseMessage]:
        """
        Select most relevant messages based on requirements and query context.
        Implements Requirements 1.4, 1.5 for message prioritization and flow preservation.
        """
        if not messages:
            return []
        
        # Always include recent messages to preserve conversation flow
        recent_count = min(requirements.required_turns * 2, len(messages))  # *2 for human+AI pairs
        recent_messages = messages[-recent_count:]
        
        # If we're within token limits, return recent messages
        if self.estimate_token_count(recent_messages) <= requirements.max_tokens:
            return recent_messages
        
        # Need to be more selective - prioritize by relevance and recency
        selected_messages = []
        current_tokens = 0
        
        # Always include the most recent exchange
        if len(messages) >= 2:
            selected_messages.extend(messages[-2:])
            current_tokens = self.estimate_token_count(messages[-2:])
        
        # Work backwards through remaining messages
        remaining_messages = messages[:-2] if len(messages) > 2 else []
        
        for i in range(len(remaining_messages) - 1, -1, -1):
            message = remaining_messages[i]
            message_tokens = self.estimate_token_count([message])
            
            if current_tokens + message_tokens > requirements.max_tokens:
                break
            
            # Calculate relevance score
            relevance_score = self._calculate_message_relevance(message, current_query)
            
            if relevance_score >= requirements.relevance_threshold:
                selected_messages.insert(-2 if len(selected_messages) >= 2 else 0, message)
                current_tokens += message_tokens
        
        return selected_messages
    
    def _calculate_message_relevance(self, message: BaseMessage, current_query: str) -> float:
        """
        Calculate relevance score between a message and current query.
        Simple keyword-based relevance scoring.
        """
        if not isinstance(message, HumanMessage):
            return 0.5  # AI messages get moderate relevance
        
        message_words = set(message.content.lower().split())
        query_words = set(current_query.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        message_words -= stop_words
        query_words -= stop_words
        
        if not query_words:
            return 0.3
        
        # Calculate Jaccard similarity
        intersection = len(message_words & query_words)
        union = len(message_words | query_words)
        
        if union == 0:
            return 0.1
        
        return intersection / union
    
    def summarize_conversation_segment(self, messages: List[BaseMessage]) -> str:
        """
        Summarize a segment of conversation messages.
        Implements Requirements 1.3 for automatic summarization of long conversations.
        """
        if not messages:
            return ""
        
        # Format messages for summarization
        conversation_text = self._format_messages_for_summary(messages)
        
        if len(conversation_text.split()) < 50:
            # Too short to summarize meaningfully
            return conversation_text
        
        summary_prompt = f"""
        Please provide a concise summary of this conversation segment. Focus on:
        1. Main topics discussed
        2. Key questions asked by the user
        3. Important information or decisions mentioned
        4. Any unresolved issues or follow-up items
        
        Keep the summary under 200 words and maintain the context for future reference.
        
        Conversation:
        {conversation_text}
        
        Summary:"""
        
        try:
            response = self._llm.invoke([HumanMessage(content=summary_prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"Summarization failed: {e}")
            # Fallback to simple truncation
            return self._create_fallback_summary(messages)
    
    def _format_messages_for_summary(self, messages: List[BaseMessage]) -> str:
        """Format messages into readable text for summarization"""
        formatted_lines = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_lines.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_lines.append(f"Assistant: {message.content}")
        
        return "\n".join(formatted_lines)
    
    def _create_fallback_summary(self, messages: List[BaseMessage]) -> str:
        """Create a simple fallback summary when LLM summarization fails"""
        topics = set()
        user_questions = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                # Extract potential topics
                content_lower = message.content.lower()
                for topic, keywords in self._topic_keywords.items():
                    if any(keyword in content_lower for keyword in keywords):
                        topics.add(topic)
                
                # Extract questions
                if any(q_word in content_lower for q_word in ["what", "how", "why", "when", "where", "which"]):
                    user_questions.append(message.content[:100] + "..." if len(message.content) > 100 else message.content)
        
        summary_parts = []
        if topics:
            summary_parts.append(f"Topics discussed: {', '.join(topics)}")
        if user_questions:
            summary_parts.append(f"Key questions: {'; '.join(user_questions[:3])}")
        
        return " | ".join(summary_parts) if summary_parts else "General conversation"
    
    def create_layered_summary(self, messages: List[BaseMessage], max_segments: int = 3) -> Dict[str, Any]:
        """
        Create a layered summary with different levels of detail.
        Implements Requirements 1.3, 1.5 for context relevance scoring and pruning.
        """
        if not messages:
            return {"full_summary": "", "segments": [], "topics": []}
        
        # Divide messages into segments
        segment_size = max(len(messages) // max_segments, 4)  # At least 4 messages per segment
        segments = []
        
        for i in range(0, len(messages), segment_size):
            segment = messages[i:i + segment_size]
            if len(segment) >= 2:  # Only process segments with meaningful content
                segments.append(segment)
        
        # Summarize each segment
        segment_summaries = []
        all_topics = set()
        
        for i, segment in enumerate(segments):
            summary = self.summarize_conversation_segment(segment)
            
            # Extract topics from this segment
            segment_topics = self._extract_topics_from_messages(segment)
            all_topics.update(segment_topics)
            
            segment_summaries.append({
                "segment_index": i,
                "message_range": (i * segment_size, min((i + 1) * segment_size, len(messages))),
                "summary": summary,
                "topics": list(segment_topics),
                "message_count": len(segment)
            })
        
        # Create overall summary
        if len(segment_summaries) > 1:
            combined_summaries = " | ".join([seg["summary"] for seg in segment_summaries])
            full_summary = f"Conversation covered: {combined_summaries}"
        else:
            full_summary = segment_summaries[0]["summary"] if segment_summaries else ""
        
        return {
            "full_summary": full_summary,
            "segments": segment_summaries,
            "topics": list(all_topics),
            "total_messages": len(messages)
        }
    
    def _extract_topics_from_messages(self, messages: List[BaseMessage]) -> set:
        """Extract topics from a set of messages"""
        topics = set()
        
        for message in messages:
            content_lower = message.content.lower()
            for topic, keywords in self._topic_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    topics.add(topic)
        
        return topics
    
    def prune_context_by_relevance(
        self, 
        messages: List[BaseMessage], 
        current_query: str,
        target_token_count: int
    ) -> Tuple[List[BaseMessage], Optional[str]]:
        """
        Prune context by relevance while maintaining conversation flow.
        Implements Requirements 1.3, 1.5 for context relevance scoring and pruning.
        """
        if not messages:
            return [], None
        
        current_tokens = self.estimate_token_count(messages)
        
        if current_tokens <= target_token_count:
            return messages, None
        
        # Create summary of older messages that will be pruned
        messages_to_summarize = []
        messages_to_keep = []
        
        # Always keep the most recent exchange
        if len(messages) >= 2:
            messages_to_keep = messages[-2:]
            remaining_tokens = target_token_count - self.estimate_token_count(messages_to_keep)
        else:
            messages_to_keep = messages
            remaining_tokens = target_token_count
        
        # Work backwards through remaining messages, keeping most relevant
        remaining_messages = messages[:-2] if len(messages) > 2 else []
        
        for i in range(len(remaining_messages) - 1, -1, -1):
            message = remaining_messages[i]
            message_tokens = self.estimate_token_count([message])
            
            if remaining_tokens >= message_tokens:
                relevance_score = self._calculate_message_relevance(message, current_query)
                
                if relevance_score >= self.config.context_relevance_threshold:
                    messages_to_keep.insert(0, message)
                    remaining_tokens -= message_tokens
                else:
                    messages_to_summarize.insert(0, message)
            else:
                messages_to_summarize.insert(0, message)
        
        # Create summary of pruned messages
        summary = None
        if messages_to_summarize:
            summary = self.summarize_conversation_segment(messages_to_summarize)
        
        return messages_to_keep, summary
    
    def get_conversation_context(
        self, 
        tenant_id: int, 
        user_id: str, 
        query: str,
        max_tokens: int = 4000
    ) -> ConversationContext:
        """
        Get optimized conversation context with prioritization and flow preservation.
        Implements Requirements 1.4, 1.5 for recent message prioritization and conversation flow preservation.
        """
        # Import here to avoid circular imports
        from rag_model.chat_data_utils import load_conversation_history
        
        # Load full conversation history
        all_messages = load_conversation_history(tenant_id, user_id)
        
        if not all_messages:
            return ConversationContext(
                recent_messages=[],
                conversation_summary=None,
                topic_context=[],
                user_preferences={},
                context_tokens=0,
                relevance_scores={},
                complexity=ConversationComplexity.SIMPLE
            )
        
        # Calculate context requirements
        requirements = self.calculate_context_requirements(query, all_messages)
        
        # Get prioritized messages
        prioritized_messages = self._prioritize_messages_with_flow_preservation(
            all_messages, query, requirements
        )
        
        # Create layered summary if needed
        conversation_summary = None
        if requirements.needs_summarization and len(all_messages) > len(prioritized_messages):
            excluded_messages = all_messages[:-len(prioritized_messages)] if prioritized_messages else all_messages
            conversation_summary = self.summarize_conversation_segment(excluded_messages)
        
        # Extract topics and user preferences
        topic_context = list(self._extract_topics_from_messages(prioritized_messages))
        user_preferences = self._extract_user_preferences(prioritized_messages)
        
        # Calculate relevance scores
        relevance_scores = {}
        for i, message in enumerate(prioritized_messages):
            relevance_scores[f"message_{i}"] = self._calculate_message_relevance(message, query)
        
        return ConversationContext(
            recent_messages=prioritized_messages,
            conversation_summary=conversation_summary,
            topic_context=topic_context,
            user_preferences=user_preferences,
            context_tokens=self.estimate_token_count(prioritized_messages),
            relevance_scores=relevance_scores,
            complexity=requirements.complexity
        )
    
    def _prioritize_messages_with_flow_preservation(
        self, 
        messages: List[BaseMessage], 
        query: str, 
        requirements: ContextRequirements
    ) -> List[BaseMessage]:
        """
        Prioritize messages while preserving conversation flow.
        Implements Requirements 1.4, 1.5 for message prioritization and flow preservation.
        """
        if not messages:
            return []
        
        # Step 1: Always preserve recent conversation flow
        recent_flow_size = min(6, len(messages))  # Last 3 exchanges (6 messages)
        recent_flow = messages[-recent_flow_size:]
        
        # Step 2: Calculate available token budget for additional context
        recent_tokens = self.estimate_token_count(recent_flow)
        remaining_token_budget = requirements.max_tokens - recent_tokens
        
        if remaining_token_budget <= 0:
            # Recent flow already exceeds budget, truncate carefully
            return self._truncate_recent_flow(recent_flow, requirements.max_tokens)
        
        # Step 3: Score and select additional messages from earlier conversation
        earlier_messages = messages[:-recent_flow_size] if len(messages) > recent_flow_size else []
        additional_messages = self._select_additional_context_messages(
            earlier_messages, query, remaining_token_budget, requirements
        )
        
        # Step 4: Combine additional context with recent flow
        return additional_messages + recent_flow
    
    def _truncate_recent_flow(self, recent_messages: List[BaseMessage], max_tokens: int) -> List[BaseMessage]:
        """Carefully truncate recent flow to fit within token budget"""
        if not recent_messages:
            return []
        
        # Always keep at least the last exchange if possible
        if len(recent_messages) >= 2:
            last_exchange = recent_messages[-2:]
            if self.estimate_token_count(last_exchange) <= max_tokens:
                return last_exchange
        
        # Fallback: keep just the last message
        if recent_messages:
            last_message = [recent_messages[-1]]
            if self.estimate_token_count(last_message) <= max_tokens:
                return last_message
        
        return []
    
    def _select_additional_context_messages(
        self, 
        earlier_messages: List[BaseMessage], 
        query: str, 
        token_budget: int,
        requirements: ContextRequirements
    ) -> List[BaseMessage]:
        """
        Select additional context messages based on relevance and importance.
        Maintains conversation flow by selecting complete exchanges when possible.
        """
        if not earlier_messages or token_budget <= 0:
            return []
        
        # Score all messages by relevance and importance
        scored_messages = []
        for i, message in enumerate(earlier_messages):
            relevance_score = self._calculate_message_relevance(message, query)
            importance_score = self._calculate_message_importance(message, i, len(earlier_messages))
            
            # Combine scores with slight preference for relevance
            combined_score = (relevance_score * 0.7) + (importance_score * 0.3)
            
            scored_messages.append({
                "message": message,
                "index": i,
                "score": combined_score,
                "tokens": self.estimate_token_count([message])
            })
        
        # Sort by score (highest first)
        scored_messages.sort(key=lambda x: x["score"], reverse=True)
        
        # Select messages that fit within token budget
        selected_messages = []
        used_tokens = 0
        selected_indices = set()
        
        for scored_msg in scored_messages:
            if used_tokens + scored_msg["tokens"] <= token_budget:
                if scored_msg["score"] >= requirements.relevance_threshold:
                    selected_messages.append(scored_msg)
                    selected_indices.add(scored_msg["index"])
                    used_tokens += scored_msg["tokens"]
        
        # Try to include conversation partners for selected messages to preserve flow
        flow_enhanced_messages = self._enhance_with_conversation_flow(
            selected_messages, earlier_messages, token_budget - used_tokens
        )
        
        # Sort by original order to maintain chronological flow
        flow_enhanced_messages.sort(key=lambda x: x["index"])
        
        return [msg["message"] for msg in flow_enhanced_messages]
    
    def _calculate_message_importance(self, message: BaseMessage, index: int, total_messages: int) -> float:
        """
        Calculate importance score for a message based on content and position.
        """
        importance_score = 0.0
        
        # Recency bonus (more recent = more important)
        recency_ratio = (index + 1) / total_messages
        importance_score += recency_ratio * 0.3
        
        # Content-based importance
        if isinstance(message, HumanMessage):
            content = message.content.lower()
            
            # Question importance
            question_indicators = ["what", "how", "why", "when", "where", "which", "who", "?"]
            if any(indicator in content for indicator in question_indicators):
                importance_score += 0.4
            
            # Problem/issue importance
            problem_indicators = ["error", "issue", "problem", "help", "stuck", "wrong", "fail"]
            if any(indicator in content for indicator in problem_indicators):
                importance_score += 0.3
            
            # Decision/conclusion importance
            decision_indicators = ["decide", "choose", "conclusion", "final", "result", "solution"]
            if any(indicator in content for indicator in decision_indicators):
                importance_score += 0.3
            
            # Length bonus for detailed messages
            word_count = len(content.split())
            if word_count > 20:
                importance_score += 0.2
            elif word_count > 10:
                importance_score += 0.1
        
        elif isinstance(message, AIMessage):
            # AI messages with detailed explanations are important
            content = message.content.lower()
            if len(content.split()) > 30:
                importance_score += 0.2
            
            # AI messages with code, examples, or structured info
            if any(indicator in content for indicator in ["```", "example", "step", "1.", "2.", "•"]):
                importance_score += 0.2
        
        return min(importance_score, 1.0)  # Cap at 1.0
    
    def _enhance_with_conversation_flow(
        self, 
        selected_messages: List[Dict], 
        all_earlier_messages: List[BaseMessage], 
        remaining_tokens: int
    ) -> List[Dict]:
        """
        Enhance selected messages by including their conversation partners when possible.
        This helps preserve conversation flow and context.
        """
        if remaining_tokens <= 0:
            return selected_messages
        
        enhanced_messages = selected_messages.copy()
        selected_indices = {msg["index"] for msg in selected_messages}
        
        # For each selected message, try to include its conversation partner
        for selected_msg in selected_messages:
            index = selected_msg["index"]
            
            # Look for conversation partner (adjacent message)
            partner_indices = []
            if index > 0 and (index - 1) not in selected_indices:
                partner_indices.append(index - 1)
            if index < len(all_earlier_messages) - 1 and (index + 1) not in selected_indices:
                partner_indices.append(index + 1)
            
            for partner_index in partner_indices:
                partner_message = all_earlier_messages[partner_index]
                partner_tokens = self.estimate_token_count([partner_message])
                
                if partner_tokens <= remaining_tokens:
                    enhanced_messages.append({
                        "message": partner_message,
                        "index": partner_index,
                        "score": 0.5,  # Lower score since it's included for flow
                        "tokens": partner_tokens
                    })
                    selected_indices.add(partner_index)
                    remaining_tokens -= partner_tokens
                    break  # Only add one partner per selected message
        
        return enhanced_messages
    
    def _extract_user_preferences(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """
        Extract user preferences from conversation history.
        """
        preferences = {
            "communication_style": "standard",
            "detail_level": "moderate",
            "topics_of_interest": [],
            "response_format": "conversational"
        }
        
        user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
        
        if not user_messages:
            return preferences
        
        # Analyze communication style
        total_words = sum(len(msg.content.split()) for msg in user_messages)
        avg_message_length = total_words / len(user_messages)
        
        if avg_message_length > 20:
            preferences["communication_style"] = "detailed"
            preferences["detail_level"] = "high"
        elif avg_message_length < 8:
            preferences["communication_style"] = "concise"
            preferences["detail_level"] = "low"
        
        # Extract topics of interest
        topic_counts = {}
        for message in user_messages:
            content_lower = message.content.lower()
            for topic, keywords in self._topic_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Get top 3 topics
        preferences["topics_of_interest"] = [
            topic for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
        
        return preferences
    
    def update_context(
        self, 
        tenant_id: int, 
        user_id: str, 
        human_message: str, 
        ai_response: str
    ) -> None:
        """
        Update conversation context with new exchange.
        Integrates with existing chat_data_utils for persistence.
        """
        # Import here to avoid circular imports
        from rag_model.chat_data_utils import append_conversation_message
        
        # Save human message
        append_conversation_message(tenant_id, user_id, "human", human_message)
        
        # Save AI response
        append_conversation_message(tenant_id, user_id, "ai", ai_response)