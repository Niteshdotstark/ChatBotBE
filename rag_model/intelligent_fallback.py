"""
Intelligent Fallback System for RAG Chatbot
Provides helpful responses when specific information isn't found in the knowledge base.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FallbackResponse:
    """Structure for fallback responses"""
    response: str
    confidence: float
    response_type: str  # "general_guidance", "procedure_help", "contact_suggestion"


class IntelligentFallbackHandler:
    """
    Provides intelligent fallback responses when specific information isn't found.
    """
    
    def __init__(self):
        self.reporting_keywords = [
            "report", "reporting", "complaint", "issue", "problem", "error", 
            "bug", "feedback", "contact", "support", "help", "submit"
        ]
        
        self.procedure_keywords = [
            "how to", "how can", "steps", "process", "procedure", "method", 
            "way to", "guide", "instructions", "tutorial"
        ]
        
        self.question_keywords = [
            "question", "exam", "test", "quiz", "assessment", "answer", "wrong"
        ]
        
        # USB and storage policy keywords
        self.usb_policy_keywords = [
            "usb", "usb drive", "flash drive", "thumb drive", "external storage", 
            "portable storage", "storage device", "removable media", "external device"
        ]
        
        self.policy_keywords = [
            "policy", "allowed", "permitted", "restricted", "approved", "can i use",
            "are usb drives", "external storage", "storage policy"
        ]
        
        # Specific policy information that should be provided
        self.policy_responses = {
            "usb_storage_policy": """
According to RAG24 Technologies' IT & Information Security Policy:

**USB Drives and External Storage:**
- USB drives and external storage are restricted unless approved
- Client data must not be copied to personal devices or cloud storage
- Confidential and Restricted data must be encrypted at rest and in transit

**Data Protection Requirements:**
- All company data is classified as Public, Internal, Confidential, or Restricted
- Personal devices and unauthorized storage are not permitted for company/client data
- Any external storage use requires IT approval

**For USB Drive Usage:**
- Contact IT Security team for approval before using any external storage
- Ensure any approved devices are encrypted and registered
- Follow data classification guidelines for all information handling

For specific approval or questions about storage devices, contact the IT Security team.
            """,
            
            "data_protection_policy": """
RAG24 Technologies' Data Protection & Privacy Policy includes:

**Key Requirements:**
- Confidential and Restricted data must be encrypted at rest and in transit
- Client data must not be copied to personal devices or cloud storage
- USB drives and external storage are restricted unless approved
- RAG24 complies with applicable data protection laws and contractual obligations

**Data Classification:**
- Public: Approved for public release
- Internal: For internal business use only  
- Confidential: Sensitive business or client data
- Restricted: Highly sensitive information with strict access controls

**Storage Restrictions:**
- Personal devices are not permitted for company data
- Cloud storage must be approved company solutions only
- External storage requires IT approval and encryption
            """
        }
        
        # Common fallback responses for different scenarios
        self.fallback_templates = {
            "reporting_procedure": """
For reporting issues or questions, here's the general approach:

**Direct Reporting (Usually Best):**
- Look for "Report" or "Report Issue" buttons in the system
- Select the type of problem you're experiencing
- Provide clear details about the issue
- Submit through the official reporting tool

**Alternative Support:**
- Contact the official support team if direct reporting isn't available
- Include specific details about what you're trying to report
- Mention the system or course name you're using

**Best Practices:**
- Be specific about the problem
- Include relevant details (but no personal info)
- Use official channels for fastest resolution
            """,
            
            "general_procedure": """
While I don't have the specific steps for that, here's a general approach:

**Common Steps:**
1. Look for relevant buttons or links in the system
2. Check official documentation or help sections
3. Contact support if you can't find the right option
4. Provide clear details when reporting issues

**Where to Look:**
- System menus or settings
- Help or support sections
- Official documentation
- Contact/support pages
            """,
            
            "contact_guidance": """
For specific procedures I don't have details on, I'd recommend:

**Official Channels:**
- Check the system's help or support section
- Look for contact information in the platform
- Use any built-in reporting or feedback tools

**When Contacting Support:**
- Be specific about what you need help with
- Include relevant details about your situation
- Mention the specific system or course you're using
            """
        }
    
    def detect_intent(self, query: str) -> Tuple[str, float]:
        """
        Detect the intent of the user's query to provide appropriate fallback.
        
        Returns:
            Tuple of (intent_type, confidence_score)
        """
        query_lower = query.lower()
        
        # Check for USB/storage policy queries first (highest priority)
        usb_matches = sum(1 for keyword in self.usb_policy_keywords if keyword in query_lower)
        policy_matches = sum(1 for keyword in self.policy_keywords if keyword in query_lower)
        
        # Check for other intents
        reporting_matches = sum(1 for keyword in self.reporting_keywords if keyword in query_lower)
        procedure_matches = sum(1 for keyword in self.procedure_keywords if keyword in query_lower)
        question_matches = sum(1 for keyword in self.question_keywords if keyword in query_lower)
        
        # Calculate confidence scores
        total_words = len(query_lower.split())
        usb_confidence = min(usb_matches / max(total_words * 0.3, 1), 1.0)
        policy_confidence = min(policy_matches / max(total_words * 0.3, 1), 1.0)
        reporting_confidence = min(reporting_matches / max(total_words * 0.3, 1), 1.0)
        procedure_confidence = min(procedure_matches / max(total_words * 0.3, 1), 1.0)
        question_confidence = min(question_matches / max(total_words * 0.3, 1), 1.0)
        
        # Determine intent based on keyword combinations
        if usb_matches > 0 and policy_matches > 0:
            return "usb_storage_policy", min(usb_confidence + policy_confidence, 1.0)
        elif usb_matches > 0:
            return "usb_storage_policy", usb_confidence
        elif "data protection" in query_lower or "privacy" in query_lower:
            return "data_protection_policy", 0.8
        elif reporting_matches > 0 and (procedure_matches > 0 or question_matches > 0):
            return "reporting_procedure", min(reporting_confidence + procedure_confidence + question_confidence, 1.0)
        elif procedure_matches > 0:
            return "general_procedure", procedure_confidence
        elif reporting_matches > 0:
            return "contact_guidance", reporting_confidence
        else:
            return "general_help", 0.3
    
    def generate_fallback_response(self, query: str, retrieved_docs: List = None) -> FallbackResponse:
        """
        Generate an intelligent fallback response when specific information isn't found.
        
        Args:
            query: The user's original query
            retrieved_docs: Any documents that were retrieved (even if not relevant)
            
        Returns:
            FallbackResponse with helpful guidance
        """
        intent, confidence = self.detect_intent(query)
        
        # Handle policy-specific responses first
        if intent in self.policy_responses:
            return FallbackResponse(
                response=self.policy_responses[intent].strip(),
                confidence=confidence,
                response_type="policy_information"
            )
        
        # Get base response template for other intents
        if intent == "reporting_procedure":
            base_response = self.fallback_templates["reporting_procedure"]
            response_type = "procedure_help"
        elif intent == "general_procedure":
            base_response = self.fallback_templates["general_procedure"]
            response_type = "general_guidance"
        else:
            base_response = self.fallback_templates["contact_guidance"]
            response_type = "contact_suggestion"
        
        # Add context-specific information if available
        if retrieved_docs and len(retrieved_docs) > 0:
            # If we have some documents but they weren't relevant enough
            response = f"I found some related information, but not the specific details you're looking for.\n\n{base_response}"
        else:
            # No relevant documents found
            response = f"I don't have specific information about that in my knowledge base.\n\n{base_response}"
        
        return FallbackResponse(
            response=response.strip(),
            confidence=confidence,
            response_type=response_type
        )
    
    def should_use_fallback(self, original_response: str, confidence_threshold: float = 0.3) -> bool:
        """
        Determine if we should use fallback instead of the original response.
        
        Args:
            original_response: The original RAG response
            confidence_threshold: Minimum confidence to use original response
            
        Returns:
            True if fallback should be used
        """
        # Check for generic "I don't know" type responses
        generic_indicators = [
            "i don't have info",
            "i couldn't find",
            "no information",
            "not mentioned",
            "doesn't mention",
            "no details",
            "not available"
        ]
        
        response_lower = original_response.lower()
        
        # If response is too short and generic, use fallback
        if len(original_response) < 100:
            for indicator in generic_indicators:
                if indicator in response_lower:
                    return True
        
        return False


# Global instance for easy access
fallback_handler = IntelligentFallbackHandler()


def enhance_response_with_fallback(query: str, original_response: str, retrieved_docs: List = None) -> str:
    """
    Enhance a response with intelligent fallback if the original response is too generic.
    
    Args:
        query: Original user query
        original_response: The RAG system's original response
        retrieved_docs: Documents that were retrieved
        
    Returns:
        Enhanced response (either original or fallback)
    """
    if fallback_handler.should_use_fallback(original_response):
        fallback = fallback_handler.generate_fallback_response(query, retrieved_docs)
        return fallback.response
    
    return original_response