"""
Adaptive Response Complexity System for RAG Chatbot
Implements user expertise detection and response complexity adjustment.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from rag_model.context_manager import ConversationContext


class ExpertiseLevel(Enum):
    """User expertise levels for response complexity adjustment"""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ResponseStyle(Enum):
    """Response style preferences"""
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    EDUCATIONAL = "educational"
    CONCISE = "concise"
    DETAILED = "detailed"


@dataclass
class ExpertiseProfile:
    """Comprehensive user expertise profile"""
    overall_level: ExpertiseLevel
    confidence: float
    domain_expertise: Dict[str, ExpertiseLevel]
    communication_style: ResponseStyle
    preferred_detail_level: str  # "minimal", "moderate", "comprehensive"
    technical_vocabulary_comfort: float
    learning_indicators: List[str]
    experience_indicators: List[str]


@dataclass
class ComplexityAdjustment:
    """Response complexity adjustment parameters"""
    vocabulary_level: str  # "simple", "intermediate", "advanced", "expert"
    explanation_depth: str  # "basic", "moderate", "detailed", "comprehensive"
    example_complexity: str  # "simple", "realistic", "advanced"
    technical_detail_level: str  # "minimal", "moderate", "high", "expert"
    assume_prior_knowledge: bool
    include_definitions: bool
    use_analogies: bool
    provide_step_by_step: bool


class AdvancedExpertiseDetector:
    """
    Advanced user expertise detection with domain-specific analysis.
    Implements Requirements 6.5 for comprehensive user expertise detection from context.
    """
    
    def __init__(self):
        # Expertise indicators by level
        self._novice_indicators = [
            "i'm new to", "just started", "don't know anything", "complete beginner",
            "never used", "first time", "what is", "how do i start"
        ]
        
        self._beginner_indicators = [
            "learning", "trying to understand", "basic question", "simple explanation",
            "help me understand", "explain like", "what does", "how does", "why does"
        ]
        
        self._intermediate_indicators = [
            "i know some", "familiar with", "used before", "understand the basics",
            "difference between", "best practice", "recommend", "which is better"
        ]
        
        self._advanced_indicators = [
            "optimize", "performance", "architecture", "design pattern", "implementation",
            "scalability", "efficiency", "trade-off", "pros and cons", "deep dive"
        ]
        
        self._expert_indicators = [
            "benchmark", "profiling", "low-level", "internals", "protocol", "specification",
            "edge case", "corner case", "optimization", "custom implementation", "research"
        ]
        
        # Domain-specific vocabularies
        self._domain_vocabularies = {
            "programming": {
                "novice": ["code", "program", "computer"],
                "beginner": ["function", "variable", "loop", "if statement"],
                "intermediate": ["class", "object", "method", "api", "library"],
                "advanced": ["algorithm", "data structure", "design pattern", "framework"],
                "expert": ["compiler", "runtime", "memory management", "concurrency"]
            },
            "data_science": {
                "novice": ["data", "analysis", "chart"],
                "beginner": ["dataset", "visualization", "statistics"],
                "intermediate": ["machine learning", "model", "training", "prediction"],
                "advanced": ["neural network", "deep learning", "feature engineering"],
                "expert": ["gradient descent", "backpropagation", "hyperparameter tuning"]
            },
            "business": {
                "novice": ["business", "company", "money"],
                "beginner": ["revenue", "profit", "customer", "market"],
                "intermediate": ["strategy", "roi", "kpi", "analytics"],
                "advanced": ["optimization", "forecasting", "segmentation"],
                "expert": ["econometric modeling", "market dynamics", "strategic positioning"]
            },
            "technology": {
                "novice": ["computer", "internet", "website"],
                "beginner": ["software", "application", "database"],
                "intermediate": ["server", "cloud", "api", "integration"],
                "advanced": ["microservices", "containerization", "devops"],
                "expert": ["distributed systems", "consensus algorithms", "fault tolerance"]
            }
        }
        
        # Communication style indicators
        self._style_indicators = {
            ResponseStyle.CONVERSATIONAL: ["chat", "talk", "discuss", "casual", "friendly"],
            ResponseStyle.TECHNICAL: ["technical", "precise", "accurate", "detailed", "specific"],
            ResponseStyle.EDUCATIONAL: ["learn", "teach", "explain", "understand", "study"],
            ResponseStyle.CONCISE: ["brief", "short", "quick", "summary", "concise"],
            ResponseStyle.DETAILED: ["detailed", "comprehensive", "thorough", "complete", "in-depth"]
        }
    
    def analyze_user_expertise(self, context: ConversationContext) -> ExpertiseProfile:
        """
        Comprehensive analysis of user expertise across multiple dimensions.
        Implements Requirements 6.5 for user expertise detection from context.
        """
        
        if not context.recent_messages:
            return self._create_default_profile()
        
        # Extract user messages for analysis
        user_messages = [
            msg.content for msg in context.recent_messages 
            if isinstance(msg, HumanMessage)
        ]
        
        if not user_messages:
            return self._create_default_profile()
        
        # Combine all user text for analysis
        combined_text = " ".join(user_messages).lower()
        
        # Analyze overall expertise level
        overall_level, confidence = self._determine_overall_expertise(combined_text, user_messages)
        
        # Analyze domain-specific expertise
        domain_expertise = self._analyze_domain_expertise(combined_text)
        
        # Determine communication style
        communication_style = self._determine_communication_style(combined_text, user_messages)
        
        # Determine preferred detail level
        detail_level = self._determine_detail_preference(combined_text, user_messages)
        
        # Calculate technical vocabulary comfort
        vocab_comfort = self._calculate_vocabulary_comfort(combined_text, domain_expertise)
        
        # Extract learning and experience indicators
        learning_indicators = self._extract_learning_indicators(combined_text)
        experience_indicators = self._extract_experience_indicators(combined_text)
        
        return ExpertiseProfile(
            overall_level=overall_level,
            confidence=confidence,
            domain_expertise=domain_expertise,
            communication_style=communication_style,
            preferred_detail_level=detail_level,
            technical_vocabulary_comfort=vocab_comfort,
            learning_indicators=learning_indicators,
            experience_indicators=experience_indicators
        )
    
    def _create_default_profile(self) -> ExpertiseProfile:
        """Create default expertise profile for new users"""
        return ExpertiseProfile(
            overall_level=ExpertiseLevel.INTERMEDIATE,
            confidence=0.5,
            domain_expertise={},
            communication_style=ResponseStyle.CONVERSATIONAL,
            preferred_detail_level="moderate",
            technical_vocabulary_comfort=0.5,
            learning_indicators=[],
            experience_indicators=[]
        )
    
    def _determine_overall_expertise(self, combined_text: str, user_messages: List[str]) -> Tuple[ExpertiseLevel, float]:
        """Determine overall expertise level with confidence score"""
        
        # Count indicators for each level
        level_scores = {
            ExpertiseLevel.NOVICE: self._count_indicators(combined_text, self._novice_indicators),
            ExpertiseLevel.BEGINNER: self._count_indicators(combined_text, self._beginner_indicators),
            ExpertiseLevel.INTERMEDIATE: self._count_indicators(combined_text, self._intermediate_indicators),
            ExpertiseLevel.ADVANCED: self._count_indicators(combined_text, self._advanced_indicators),
            ExpertiseLevel.EXPERT: self._count_indicators(combined_text, self._expert_indicators)
        }
        
        # Analyze message characteristics
        avg_message_length = sum(len(msg.split()) for msg in user_messages) / len(user_messages)
        question_complexity = self._analyze_question_complexity(user_messages)
        technical_term_density = self._calculate_technical_term_density(combined_text)
        
        # Adjust scores based on characteristics
        if avg_message_length > 25:
            level_scores[ExpertiseLevel.ADVANCED] += 2
            level_scores[ExpertiseLevel.EXPERT] += 1
        elif avg_message_length < 8:
            level_scores[ExpertiseLevel.NOVICE] += 1
            level_scores[ExpertiseLevel.BEGINNER] += 1
        
        if question_complexity > 0.7:
            level_scores[ExpertiseLevel.ADVANCED] += 2
            level_scores[ExpertiseLevel.EXPERT] += 2
        elif question_complexity < 0.3:
            level_scores[ExpertiseLevel.NOVICE] += 1
            level_scores[ExpertiseLevel.BEGINNER] += 1
        
        if technical_term_density > 0.15:
            level_scores[ExpertiseLevel.ADVANCED] += 3
            level_scores[ExpertiseLevel.EXPERT] += 2
        elif technical_term_density < 0.05:
            level_scores[ExpertiseLevel.NOVICE] += 2
            level_scores[ExpertiseLevel.BEGINNER] += 1
        
        # Find the level with highest score
        max_level = max(level_scores.items(), key=lambda x: x[1])
        detected_level = max_level[0]
        
        # Calculate confidence
        total_score = sum(level_scores.values())
        confidence = max_level[1] / max(total_score, 1) if total_score > 0 else 0.5
        
        # Boost confidence based on clear indicators
        if max_level[1] >= 3:
            confidence = min(confidence * 1.3, 1.0)
        
        return detected_level, confidence
    
    def _analyze_domain_expertise(self, combined_text: str) -> Dict[str, ExpertiseLevel]:
        """Analyze expertise in specific domains"""
        
        domain_expertise = {}
        
        for domain, vocab_levels in self._domain_vocabularies.items():
            domain_score = 0
            level_scores = {}
            
            for level, terms in vocab_levels.items():
                term_count = sum(1 for term in terms if term in combined_text)
                level_scores[level] = term_count
                domain_score += term_count
            
            if domain_score > 0:
                # Find the highest scoring level for this domain
                max_level_name = max(level_scores.items(), key=lambda x: x[1])[0]
                
                # Convert string to enum
                level_mapping = {
                    "novice": ExpertiseLevel.NOVICE,
                    "beginner": ExpertiseLevel.BEGINNER,
                    "intermediate": ExpertiseLevel.INTERMEDIATE,
                    "advanced": ExpertiseLevel.ADVANCED,
                    "expert": ExpertiseLevel.EXPERT
                }
                
                domain_expertise[domain] = level_mapping[max_level_name]
        
        return domain_expertise
    
    def _determine_communication_style(self, combined_text: str, user_messages: List[str]) -> ResponseStyle:
        """Determine preferred communication style"""
        
        style_scores = {}
        
        for style, indicators in self._style_indicators.items():
            score = sum(1 for indicator in indicators if indicator in combined_text)
            style_scores[style] = score
        
        # Analyze message patterns
        avg_length = sum(len(msg.split()) for msg in user_messages) / len(user_messages)
        
        # Adjust scores based on message characteristics
        if avg_length > 20:
            style_scores[ResponseStyle.DETAILED] += 2
            style_scores[ResponseStyle.TECHNICAL] += 1
        elif avg_length < 8:
            style_scores[ResponseStyle.CONCISE] += 2
            style_scores[ResponseStyle.CONVERSATIONAL] += 1
        
        # Check for question patterns
        question_patterns = ["how", "what", "why", "explain", "help me understand"]
        if any(pattern in combined_text for pattern in question_patterns):
            style_scores[ResponseStyle.EDUCATIONAL] += 2
        
        # Return style with highest score, default to conversational
        if not style_scores or max(style_scores.values()) == 0:
            return ResponseStyle.CONVERSATIONAL
        
        return max(style_scores.items(), key=lambda x: x[1])[0]
    
    def _determine_detail_preference(self, combined_text: str, user_messages: List[str]) -> str:
        """Determine preferred level of detail in responses"""
        
        detail_indicators = {
            "minimal": ["brief", "short", "quick", "summary", "just tell me"],
            "moderate": ["explain", "help me understand", "how does"],
            "comprehensive": ["detailed", "thorough", "complete", "in-depth", "everything about"]
        }
        
        detail_scores = {}
        for level, indicators in detail_indicators.items():
            score = sum(1 for indicator in indicators if indicator in combined_text)
            detail_scores[level] = score
        
        # Analyze message length as indicator
        avg_length = sum(len(msg.split()) for msg in user_messages) / len(user_messages)
        
        if avg_length > 25:
            detail_scores["comprehensive"] += 2
        elif avg_length < 8:
            detail_scores["minimal"] += 2
        else:
            detail_scores["moderate"] += 1
        
        # Return preference with highest score
        if not detail_scores or max(detail_scores.values()) == 0:
            return "moderate"
        
        return max(detail_scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_vocabulary_comfort(self, combined_text: str, domain_expertise: Dict[str, ExpertiseLevel]) -> float:
        """Calculate comfort level with technical vocabulary"""
        
        if not domain_expertise:
            return 0.5
        
        # Calculate average expertise level across domains
        level_values = {
            ExpertiseLevel.NOVICE: 0.1,
            ExpertiseLevel.BEGINNER: 0.3,
            ExpertiseLevel.INTERMEDIATE: 0.5,
            ExpertiseLevel.ADVANCED: 0.8,
            ExpertiseLevel.EXPERT: 1.0
        }
        
        total_comfort = sum(level_values[level] for level in domain_expertise.values())
        avg_comfort = total_comfort / len(domain_expertise)
        
        # Adjust based on technical term usage
        technical_density = self._calculate_technical_term_density(combined_text)
        adjusted_comfort = (avg_comfort + technical_density) / 2
        
        return min(adjusted_comfort, 1.0)
    
    def _calculate_technical_term_density(self, text: str) -> float:
        """Calculate density of technical terms in text"""
        
        all_technical_terms = []
        for domain_vocab in self._domain_vocabularies.values():
            for level_terms in domain_vocab.values():
                all_technical_terms.extend(level_terms)
        
        words = text.split()
        if not words:
            return 0.0
        
        technical_count = sum(1 for word in words if word in all_technical_terms)
        return technical_count / len(words)
    
    def _analyze_question_complexity(self, user_messages: List[str]) -> float:
        """Analyze the complexity of questions asked by the user"""
        
        complexity_indicators = [
            "how does", "why does", "what happens when", "difference between",
            "compare", "analyze", "optimize", "implement", "design"
        ]
        
        simple_indicators = [
            "what is", "who is", "when", "where", "yes or no", "true or false"
        ]
        
        total_complexity = 0
        total_questions = 0
        
        for message in user_messages:
            message_lower = message.lower()
            
            if "?" in message or any(q in message_lower for q in ["what", "how", "why", "when", "where"]):
                total_questions += 1
                
                complexity_score = 0
                complexity_score += sum(1 for indicator in complexity_indicators if indicator in message_lower)
                complexity_score -= sum(0.5 for indicator in simple_indicators if indicator in message_lower)
                
                # Adjust for message length
                word_count = len(message.split())
                if word_count > 15:
                    complexity_score += 1
                elif word_count < 5:
                    complexity_score -= 0.5
                
                total_complexity += max(complexity_score, 0)
        
        if total_questions == 0:
            return 0.5
        
        avg_complexity = total_complexity / total_questions
        return min(avg_complexity / 3, 1.0)  # Normalize to 0-1 range
    
    def _extract_learning_indicators(self, text: str) -> List[str]:
        """Extract indicators that the user is in learning mode"""
        
        learning_phrases = [
            "learning", "studying", "trying to understand", "new to",
            "help me learn", "teach me", "explain", "don't understand"
        ]
        
        found_indicators = []
        for phrase in learning_phrases:
            if phrase in text:
                found_indicators.append(phrase)
        
        return found_indicators[:5]  # Limit to top 5
    
    def _extract_experience_indicators(self, text: str) -> List[str]:
        """Extract indicators of user's experience level"""
        
        experience_phrases = [
            "i've used", "i know", "familiar with", "experienced with",
            "worked with", "implemented", "built", "developed"
        ]
        
        found_indicators = []
        for phrase in experience_phrases:
            if phrase in text:
                found_indicators.append(phrase)
        
        return found_indicators[:5]  # Limit to top 5
    
    def _count_indicators(self, text: str, indicators: List[str]) -> int:
        """Count how many indicators are present in the text"""
        return sum(1 for indicator in indicators if indicator in text)


class ResponseComplexityAdjuster:
    """
    Adjusts response complexity based on user expertise profile.
    Implements Requirements 6.5 for response complexity adjustment.
    """
    
    def __init__(self):
        self.complexity_templates = self._create_complexity_templates()
    
    def _create_complexity_templates(self) -> Dict[ExpertiseLevel, ComplexityAdjustment]:
        """Create complexity adjustment templates for each expertise level"""
        
        return {
            ExpertiseLevel.NOVICE: ComplexityAdjustment(
                vocabulary_level="simple",
                explanation_depth="basic",
                example_complexity="simple",
                technical_detail_level="minimal",
                assume_prior_knowledge=False,
                include_definitions=True,
                use_analogies=True,
                provide_step_by_step=True
            ),
            ExpertiseLevel.BEGINNER: ComplexityAdjustment(
                vocabulary_level="simple",
                explanation_depth="moderate",
                example_complexity="simple",
                technical_detail_level="minimal",
                assume_prior_knowledge=False,
                include_definitions=True,
                use_analogies=True,
                provide_step_by_step=True
            ),
            ExpertiseLevel.INTERMEDIATE: ComplexityAdjustment(
                vocabulary_level="intermediate",
                explanation_depth="moderate",
                example_complexity="realistic",
                technical_detail_level="moderate",
                assume_prior_knowledge=True,
                include_definitions=False,
                use_analogies=False,
                provide_step_by_step=False
            ),
            ExpertiseLevel.ADVANCED: ComplexityAdjustment(
                vocabulary_level="advanced",
                explanation_depth="detailed",
                example_complexity="advanced",
                technical_detail_level="high",
                assume_prior_knowledge=True,
                include_definitions=False,
                use_analogies=False,
                provide_step_by_step=False
            ),
            ExpertiseLevel.EXPERT: ComplexityAdjustment(
                vocabulary_level="expert",
                explanation_depth="comprehensive",
                example_complexity="advanced",
                technical_detail_level="expert",
                assume_prior_knowledge=True,
                include_definitions=False,
                use_analogies=False,
                provide_step_by_step=False
            )
        }
    
    def determine_response_complexity(self, expertise_profile: ExpertiseProfile) -> ComplexityAdjustment:
        """
        Determine appropriate response complexity based on user expertise profile.
        Implements Requirements 6.5 for response complexity adjustment.
        """
        
        # Start with base template for overall expertise level
        base_adjustment = self.complexity_templates[expertise_profile.overall_level]
        
        # Create a copy to modify
        adjusted = ComplexityAdjustment(
            vocabulary_level=base_adjustment.vocabulary_level,
            explanation_depth=base_adjustment.explanation_depth,
            example_complexity=base_adjustment.example_complexity,
            technical_detail_level=base_adjustment.technical_detail_level,
            assume_prior_knowledge=base_adjustment.assume_prior_knowledge,
            include_definitions=base_adjustment.include_definitions,
            use_analogies=base_adjustment.use_analogies,
            provide_step_by_step=base_adjustment.provide_step_by_step
        )
        
        # Adjust based on communication style preference
        if expertise_profile.communication_style == ResponseStyle.CONCISE:
            adjusted.explanation_depth = self._reduce_depth(adjusted.explanation_depth)
            adjusted.provide_step_by_step = False
        elif expertise_profile.communication_style == ResponseStyle.DETAILED:
            adjusted.explanation_depth = self._increase_depth(adjusted.explanation_depth)
            adjusted.provide_step_by_step = True
        elif expertise_profile.communication_style == ResponseStyle.EDUCATIONAL:
            adjusted.include_definitions = True
            adjusted.use_analogies = True
            adjusted.provide_step_by_step = True
        
        # Adjust based on detail preference
        if expertise_profile.preferred_detail_level == "minimal":
            adjusted.explanation_depth = "basic"
            adjusted.technical_detail_level = "minimal"
        elif expertise_profile.preferred_detail_level == "comprehensive":
            adjusted.explanation_depth = "comprehensive"
            adjusted.technical_detail_level = self._increase_technical_level(adjusted.technical_detail_level)
        
        # Adjust based on technical vocabulary comfort
        if expertise_profile.technical_vocabulary_comfort < 0.3:
            adjusted.vocabulary_level = "simple"
            adjusted.include_definitions = True
        elif expertise_profile.technical_vocabulary_comfort > 0.8:
            adjusted.vocabulary_level = self._increase_vocabulary_level(adjusted.vocabulary_level)
        
        # Adjust based on learning indicators
        if expertise_profile.learning_indicators:
            adjusted.use_analogies = True
            adjusted.provide_step_by_step = True
            adjusted.include_definitions = True
        
        return adjusted
    
    def _reduce_depth(self, current_depth: str) -> str:
        """Reduce explanation depth by one level"""
        depth_levels = ["basic", "moderate", "detailed", "comprehensive"]
        try:
            current_index = depth_levels.index(current_depth)
            return depth_levels[max(0, current_index - 1)]
        except ValueError:
            return "basic"
    
    def _increase_depth(self, current_depth: str) -> str:
        """Increase explanation depth by one level"""
        depth_levels = ["basic", "moderate", "detailed", "comprehensive"]
        try:
            current_index = depth_levels.index(current_depth)
            return depth_levels[min(len(depth_levels) - 1, current_index + 1)]
        except ValueError:
            return "moderate"
    
    def _increase_technical_level(self, current_level: str) -> str:
        """Increase technical detail level by one level"""
        tech_levels = ["minimal", "moderate", "high", "expert"]
        try:
            current_index = tech_levels.index(current_level)
            return tech_levels[min(len(tech_levels) - 1, current_index + 1)]
        except ValueError:
            return "moderate"
    
    def _increase_vocabulary_level(self, current_level: str) -> str:
        """Increase vocabulary level by one level"""
        vocab_levels = ["simple", "intermediate", "advanced", "expert"]
        try:
            current_index = vocab_levels.index(current_level)
            return vocab_levels[min(len(vocab_levels) - 1, current_index + 1)]
        except ValueError:
            return "intermediate"
    
    def create_complexity_prompt_additions(self, adjustment: ComplexityAdjustment) -> str:
        """Create prompt additions based on complexity adjustment"""
        
        prompt_parts = []
        
        # Vocabulary level guidance
        vocab_guidance = {
            "simple": "Use simple, everyday language. Avoid jargon and technical terms.",
            "intermediate": "Use clear language with some technical terms, but explain them when first used.",
            "advanced": "Use appropriate technical vocabulary, assuming familiarity with domain concepts.",
            "expert": "Use precise technical language and domain-specific terminology freely."
        }
        prompt_parts.append(f"Language Level: {vocab_guidance[adjustment.vocabulary_level]}")
        
        # Explanation depth guidance
        depth_guidance = {
            "basic": "Provide brief, straightforward explanations focusing on the essential points.",
            "moderate": "Provide balanced explanations with key details and context.",
            "detailed": "Provide thorough explanations with comprehensive details and background.",
            "comprehensive": "Provide exhaustive explanations covering all aspects and nuances."
        }
        prompt_parts.append(f"Explanation Depth: {depth_guidance[adjustment.explanation_depth]}")
        
        # Technical detail guidance
        tech_guidance = {
            "minimal": "Minimize technical details, focus on practical outcomes.",
            "moderate": "Include relevant technical details that aid understanding.",
            "high": "Provide significant technical depth and implementation details.",
            "expert": "Include advanced technical details, edge cases, and optimization considerations."
        }
        prompt_parts.append(f"Technical Detail: {tech_guidance[adjustment.technical_detail_level]}")
        
        # Additional instructions
        if adjustment.include_definitions:
            prompt_parts.append("Include definitions for technical terms and concepts.")
        
        if adjustment.use_analogies:
            prompt_parts.append("Use analogies and metaphors to explain complex concepts.")
        
        if adjustment.provide_step_by_step:
            prompt_parts.append("Break down processes into clear, step-by-step instructions.")
        
        if not adjustment.assume_prior_knowledge:
            prompt_parts.append("Do not assume prior knowledge; explain foundational concepts.")
        
        return "\n".join(prompt_parts)


# Integration functions
def detect_user_expertise(context: ConversationContext) -> ExpertiseProfile:
    """
    Main function to detect user expertise from conversation context.
    Implements Requirements 6.5 for user expertise detection.
    """
    detector = AdvancedExpertiseDetector()
    return detector.analyze_user_expertise(context)


def adjust_response_complexity(expertise_profile: ExpertiseProfile) -> ComplexityAdjustment:
    """
    Main function to determine response complexity adjustments.
    Implements Requirements 6.5 for response complexity adjustment.
    """
    adjuster = ResponseComplexityAdjuster()
    return adjuster.determine_response_complexity(expertise_profile)