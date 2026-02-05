"""
Response Summarization Module
Provides functionality to summarize detailed RAG responses for social media platforms
"""

import re
from typing import Dict, Any, Optional
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import os

class ResponseSummarizer:
    """
    Summarizes detailed RAG responses for different platforms and contexts
    """
    
    def __init__(self, model_id: str = "meta.llama3-8b-instruct-v1:0", region_name: str = "ap-south-1"):
        self.model_id = model_id
        self.region_name = region_name
        
        # Initialize LLM for summarization
        self._llm = ChatBedrock(
            model_id=model_id,
            region_name=region_name,
            model_kwargs={"temperature": 0.2}  # Lower temperature for consistent summaries
        )
        
        # Create summarization prompt template
        self._create_summarization_prompt()
    
    def _create_summarization_prompt(self):
        """Create the prompt template for summarization"""
        
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a text summarizer. Create a brief, actionable summary (under 1000 characters) of the given text for social media. Focus on key actions and options. Use simple language and be concise. DO NOT include hashtags, emojis, or social media formatting. Provide only clean, professional text."),
            HumanMessagePromptTemplate.from_template("{detailed_response}\n\nSummarize this in under 1000 characters without hashtags:")
        ])
    
    def summarize_response(
        self, 
        detailed_response: str, 
        max_length: int = 1000,  # Changed default to 1000 characters
        platform: str = "social_media"
    ) -> Dict[str, Any]:
        """
        Summarize a detailed response for social media platforms
        
        Args:
            detailed_response: The full, detailed response to summarize
            max_length: Maximum character length for the summary
            platform: Platform type (social_media, telegram, etc.)
            
        Returns:
            Dict containing summary and metadata
        """
        
        try:
            # Generate summary using LLM
            messages = self.summarization_prompt.format_messages(
                detailed_response=detailed_response
            )
            
            response = self._llm.invoke(messages)
            
            summary = response.content.strip()
            
            # Post-process summary based on platform and length requirements
            final_summary = self._post_process_summary(summary, max_length, platform)
            
            return {
                "summary": final_summary,
                "original_length": len(detailed_response),
                "summary_length": len(final_summary),
                "compression_ratio": len(final_summary) / len(detailed_response),
                "platform": platform,
                "success": True
            }
            
        except Exception as e:
            print(f"Summarization failed: {e}")
            # Fallback to simple truncation
            fallback_summary = self._create_fallback_summary(detailed_response, max_length)
            
            return {
                "summary": fallback_summary,
                "original_length": len(detailed_response),
                "summary_length": len(fallback_summary),
                "compression_ratio": len(fallback_summary) / len(detailed_response),
                "platform": platform,
                "success": False,
                "error": str(e)
            }
    
    def _post_process_summary(self, summary: str, max_length: int, platform: str) -> str:
        """Post-process the generated summary"""
        
        # Remove any quotes that might have been added
        summary = summary.strip('"\'')
        
        # Remove hashtags and social media formatting
        import re
        summary = re.sub(r'#\w+', '', summary)  # Remove hashtags like #AWS #ExamQuestions
        summary = re.sub(r'\s+', ' ', summary)  # Clean up extra spaces
        summary = summary.strip()
        
        # Ensure it fits within length limits
        if len(summary) > max_length:
            # Try to truncate at sentence boundary
            sentences = re.split(r'[.!?]+', summary)
            truncated = ""
            
            for sentence in sentences:
                if len(truncated + sentence + ".") <= max_length - 3:  # Leave room for "..."
                    truncated += sentence + "."
                else:
                    break
            
            if truncated:
                summary = truncated.rstrip(".") + "..."
            else:
                # Hard truncation as last resort
                summary = summary[:max_length-3] + "..."
        
        # Platform-specific adjustments
        if platform == "telegram":
            # Telegram supports more formatting
            pass
        elif platform == "facebook" or platform == "instagram":
            # Social media platforms - keep it very concise
            if len(summary) > 250:
                summary = summary[:247] + "..."
        
        return summary
    
    def _create_fallback_summary(self, detailed_response: str, max_length: int) -> str:
        """Create a simple fallback summary when LLM fails"""
        
        # Extract first sentence or paragraph
        sentences = re.split(r'[.!?]+', detailed_response)
        
        if sentences and len(sentences[0]) <= max_length:
            return sentences[0].strip() + "."
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(detailed_response)
        
        if key_phrases:
            summary = " ".join(key_phrases)
            if len(summary) <= max_length:
                return summary
        
        # Last resort: simple truncation
        if len(detailed_response) <= max_length:
            return detailed_response
        
        return detailed_response[:max_length-3] + "..."
    
    def _extract_key_phrases(self, text: str) -> list:
        """Extract key phrases from text for fallback summary"""
        
        # Look for structured content
        key_indicators = [
            r"Option \d+:",
            r"Step \d+:",
            r"\*\*[^*]+\*\*",  # Bold text
            r"Important:",
            r"Note:",
            r"Best practice:",
        ]
        
        key_phrases = []
        
        for pattern in key_indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_phrases.extend(matches[:2])  # Limit to 2 matches per pattern
        
        return key_phrases[:3]  # Return top 3 key phrases


def create_dual_response(
    detailed_response: str, 
    sources: list, 
    response_mode: str = "detailed"
) -> Dict[str, Any]:
    """
    Create both detailed and summarized responses
    
    Args:
        detailed_response: The full detailed response
        sources: List of sources
        response_mode: "detailed", "summary", or "both"
        
    Returns:
        Dict with appropriate response based on mode
    """
    
    if response_mode == "detailed":
        return {
            "response": detailed_response,
            "sources": sources,
            "response_type": "detailed"
        }
    
    elif response_mode == "summary":
        summarizer = ResponseSummarizer()
        summary_result = summarizer.summarize_response(detailed_response)
        
        return {
            "response": summary_result["summary"],
            "sources": sources[:2],  # Limit sources for summary
            "response_type": "summary",
            "summary_metadata": {
                "original_length": summary_result["original_length"],
                "compression_ratio": summary_result["compression_ratio"]
            }
        }
    
    elif response_mode == "both":
        summarizer = ResponseSummarizer()
        summary_result = summarizer.summarize_response(detailed_response)
        
        return {
            "detailed_response": detailed_response,
            "summary_response": summary_result["summary"],
            "sources": sources,
            "response_type": "both",
            "summary_metadata": {
                "original_length": summary_result["original_length"],
                "compression_ratio": summary_result["compression_ratio"]
            }
        }
    
    else:
        raise ValueError(f"Invalid response_mode: {response_mode}")


# Convenience function for backward compatibility
def summarize_for_social_media(detailed_response: str, platform: str = "social_media") -> str:
    """
    Quick function to summarize a response for social media with 1000 character limit
    
    Args:
        detailed_response: The detailed response to summarize
        platform: Platform type (social_media, telegram, etc.)
        
    Returns:
        Summarized response string (max 1000 characters)
    """
    
    summarizer = ResponseSummarizer()
    result = summarizer.summarize_response(detailed_response, max_length=1000, platform=platform)
    return result["summary"]


if __name__ == "__main__":
    # Test the summarizer
    test_response = """
    To report an AWS exam question, you have the following options:

    **Option 1: Report directly from the question screen (RECOMMENDED)**
    1. While viewing the question, look for and click "Report Content Errors" button
    2. Select the type of issue from the dropdown menu

    **Option 2: Contact AWS Training & Certification Support**
    1. Go to AWS Training & Certification support website
    2. Set up your support request with inquiry type: Certification

    **Best Practices for Reporting**
    1. Be Specific: Provide detailed descriptions rather than vague complaints
    2. Include Context: Mention the course, module, or question number if available
    """
    
    summarizer = ResponseSummarizer()
    result = summarizer.summarize_response(test_response)
    
    print("Original length:", result["original_length"])
    print("Summary length:", result["summary_length"])
    print("Compression ratio:", f"{result['compression_ratio']:.2%}")
    print("Summary:", result["summary"])