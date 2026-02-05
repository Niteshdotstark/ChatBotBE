#!/usr/bin/env python3
"""
Intelligent Query Processor that makes the chatbot "smarter" by:
1. Expanding queries to find related content
2. Using semantic similarity for better matching
3. Applying reasoning to connect concepts
"""

import re
import boto3
from typing import List, Dict, Any, Optional, Tuple
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

class IntelligentQueryProcessor:
    """Processes queries intelligently to improve retrieval accuracy"""
    
    def __init__(self, region_name: str = "ap-south-1"):
        self.region_name = region_name
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region_name)
        
        # Use a fast model for query processing
        self.llm = ChatBedrock(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",  # Fast and efficient
            region_name=region_name,
            model_kwargs={"temperature": 0.1, "max_tokens": 200}
        )
        
        # Query expansion patterns
        self.expansion_patterns = {
            # Time-based queries
            r"day\s*(\d+)": ["day {}", "day {} process", "day {} onboarding", "day {} activities", "{} day"],
            r"what happens on": ["process", "procedure", "steps", "activities", "workflow"],
            r"first day": ["day 1", "day one", "onboarding", "orientation", "initial"],
            
            # Process queries  
            r"onboarding": ["orientation", "new employee", "first day", "joining process"],
            r"process": ["procedure", "steps", "workflow", "activities", "how to"],
            
            # Skills/Experience queries
            r"skills": ["expertise", "experience", "proficient", "knowledge", "abilities"],
            r"projects": ["work", "developed", "built", "created", "experience"],
            r"experience": ["worked", "employment", "background", "history"],
            
            # AWS specific
            r"aws": ["amazon web services", "cloud", "ec2", "s3", "lambda"],
            r"certification": ["exam", "test", "certified", "qualification"],
            
            # USB and Storage specific - ENHANCED
            r"usb\s*drive": ["USB drives", "external storage", "portable storage", "flash drive", "thumb drive", "storage device", "removable media"],
            r"usb": ["USB drives", "external storage", "portable storage", "removable storage", "storage policy", "device policy"],
            r"external\s*storage": ["USB drives", "portable storage", "removable media", "storage devices", "external devices", "storage policy"],
            r"can\s*i\s*use": ["policy", "allowed", "permitted", "restrictions", "guidelines", "rules"],
            r"drive\s*policy": ["storage policy", "USB policy", "device policy", "external storage", "removable media policy"],
            r"storage\s*policy": ["USB policy", "external storage", "device policy", "removable media", "portable storage"],
            
            # Data protection and security
            r"data\s*protection": ["information security", "IT security", "privacy policy", "confidential data", "restricted data"],
            r"security\s*policy": ["IT policy", "information security", "data protection", "access control"],
        }
        
        # Context clues for better understanding
        self.context_clues = {
            "day": ["onboarding", "process", "orientation", "training"],
            "first": ["initial", "beginning", "start", "new"],
            "what happens": ["process", "procedure", "steps", "workflow"],
            "how": ["process", "method", "way", "procedure"],
            
            # USB and storage context clues - ENHANCED
            "usb": ["policy", "allowed", "restricted", "approved", "security", "external storage"],
            "drive": ["policy", "storage", "external", "portable", "removable", "security"],
            "external": ["storage", "policy", "restricted", "approved", "security", "USB"],
            "storage": ["policy", "external", "USB", "restricted", "approved", "security"],
            "can i use": ["policy", "allowed", "permitted", "restricted", "approved"],
            "allowed": ["policy", "permitted", "restricted", "approved", "security"],
            "policy": ["security", "IT", "information", "data protection", "guidelines"],
            "restricted": ["policy", "approved", "security", "unauthorized", "prohibited"],
            "approved": ["policy", "restricted", "security", "authorized", "permitted"],
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with related terms and variations"""
        query_lower = query.lower().strip()
        expanded_queries = [query]  # Always include original
        
        # Apply expansion patterns
        for pattern, expansions in self.expansion_patterns.items():
            matches = re.findall(pattern, query_lower)
            if matches:
                for match in matches:
                    for expansion in expansions:
                        if "{}" in expansion:
                            expanded_query = expansion.format(match)
                        else:
                            expanded_query = f"{query_lower} {expansion}"
                        expanded_queries.append(expanded_query)
        
        # Add context-based expansions
        for clue, related_terms in self.context_clues.items():
            if clue in query_lower:
                for term in related_terms:
                    expanded_queries.append(f"{query_lower} {term}")
                    expanded_queries.append(f"{term} {query_lower}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in expanded_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries[:5]  # Limit to top 5 to avoid noise
    
    def generate_semantic_variations(self, query: str) -> List[str]:
        """Generate semantic variations using LLM"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""
You are a query expansion expert. Given a user query, generate 3-4 alternative ways to ask the same question that might match different document phrasings.

Rules:
1. Keep the core meaning identical
2. Use different words/phrases that mean the same thing
3. Consider formal vs informal language
4. Think about how the information might be written in documents
5. Be concise - each variation should be 1-2 sentences max

Example:
Query: "What happens on Day 1?"
Variations:
- Day 1 process
- First day activities  
- Day one onboarding
- Initial day procedures
"""),
                HumanMessagePromptTemplate.from_template("Query: {query}\n\nVariations:")
            ])
            
            response = self.llm.invoke(prompt.format_messages(query=query))
            variations_text = response.content.strip()
            
            # Parse variations from response
            variations = []
            for line in variations_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('Query:') and not line.startswith('Variations:'):
                    # Remove bullet points and numbering
                    clean_line = re.sub(r'^[-*•\d.)\s]+', '', line).strip()
                    if clean_line and len(clean_line) > 3:
                        variations.append(clean_line)
            
            return variations[:4]  # Limit to 4 variations
            
        except Exception as e:
            print(f"Error generating semantic variations: {e}")
            return []
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand user intent"""
        query_lower = query.lower().strip()
        
        intent_analysis = {
            "query_type": "general",
            "entities": [],
            "time_references": [],
            "action_words": [],
            "specificity": "medium"
        }
        
        # Detect query type
        if any(word in query_lower for word in ["what", "how", "when", "where", "why"]):
            intent_analysis["query_type"] = "question"
        elif any(word in query_lower for word in ["list", "show", "tell me about"]):
            intent_analysis["query_type"] = "information_request"
        elif any(word in query_lower for word in ["explain", "describe"]):
            intent_analysis["query_type"] = "explanation"
        
        # Extract entities
        entities = []
        if "day" in query_lower:
            day_matches = re.findall(r"day\s*(\d+|one|1)", query_lower)
            if day_matches:
                entities.extend([f"day {match}" for match in day_matches])
        
        # Extract time references
        time_refs = re.findall(r"(day\s*\d+|first\s+day|initial|beginning|start)", query_lower)
        intent_analysis["time_references"] = time_refs
        
        # Extract action words
        action_words = re.findall(r"(happens|process|procedure|steps|activities|onboarding)", query_lower)
        intent_analysis["action_words"] = action_words
        
        # Determine specificity
        if len(query.split()) <= 3:
            intent_analysis["specificity"] = "low"
        elif len(query.split()) >= 8:
            intent_analysis["specificity"] = "high"
        
        intent_analysis["entities"] = entities
        return intent_analysis
    
    def process_query(self, original_query: str) -> Dict[str, Any]:
        """Main processing function that enhances the query"""
        print(f"🧠 Processing query: '{original_query}'")
        
        # Step 1: Analyze intent
        intent = self.analyze_query_intent(original_query)
        print(f"📊 Intent analysis: {intent}")
        
        # Step 2: Expand query with patterns
        expanded_queries = self.expand_query(original_query)
        print(f"🔍 Pattern expansions: {expanded_queries}")
        
        # Step 3: Generate semantic variations
        semantic_variations = self.generate_semantic_variations(original_query)
        print(f"🎯 Semantic variations: {semantic_variations}")
        
        # Step 4: Combine all queries
        all_queries = [original_query] + expanded_queries + semantic_variations
        
        # Remove duplicates and rank by relevance
        unique_queries = []
        seen = set()
        for query in all_queries:
            query_clean = query.lower().strip()
            if query_clean not in seen and len(query_clean) > 2:
                seen.add(query_clean)
                unique_queries.append(query)
        
        return {
            "original_query": original_query,
            "enhanced_queries": unique_queries[:8],  # Top 8 queries
            "intent_analysis": intent,
            "processing_strategy": self._determine_strategy(intent)
        }
    
    def _determine_strategy(self, intent: Dict[str, Any]) -> str:
        """Determine the best retrieval strategy based on intent"""
        if intent["specificity"] == "low":
            return "broad_search"  # Use multiple query variations
        elif intent["query_type"] == "question":
            return "semantic_search"  # Focus on semantic similarity
        elif len(intent["entities"]) > 0:
            return "entity_focused"  # Focus on specific entities
        else:
            return "standard"

def enhance_retrieval_with_intelligent_processing(
    query: str, 
    tenant_id: int, 
    retrieval_function,
    top_k: int = 5
) -> Tuple[List[Document], Dict[str, Any]]:
    """Enhanced retrieval using intelligent query processing"""
    
    processor = IntelligentQueryProcessor()
    
    # Process the query
    query_analysis = processor.process_query(query)
    
    # Retrieve documents using multiple query variations
    all_documents = []
    query_results = {}
    
    for enhanced_query in query_analysis["enhanced_queries"]:
        try:
            docs = retrieval_function(enhanced_query, tenant_id, top_k)
            if docs:
                all_documents.extend(docs)
                query_results[enhanced_query] = len(docs)
        except Exception as e:
            print(f"Error retrieving for query '{enhanced_query}': {e}")
            query_results[enhanced_query] = 0
    
    # Remove duplicates based on content similarity
    unique_documents = []
    seen_content = set()
    
    for doc in all_documents:
        # Use first 200 characters as content signature
        content_signature = doc.page_content[:200].lower().strip()
        if content_signature not in seen_content:
            seen_content.add(content_signature)
            unique_documents.append(doc)
    
    # Sort by relevance (this could be enhanced with scoring)
    final_documents = unique_documents[:top_k]
    
    retrieval_metadata = {
        "query_analysis": query_analysis,
        "query_results": query_results,
        "total_documents_found": len(all_documents),
        "unique_documents": len(unique_documents),
        "final_documents": len(final_documents)
    }
    
    print(f"🎯 Intelligent retrieval: {len(final_documents)} final documents")
    return final_documents, retrieval_metadata

# Integration function for existing system
def answer_question_with_intelligent_processing(
    question: str, 
    tenant_id: int, 
    user_id: str = "default",
    context_messages: list = None,
    response_mode: str = "detailed"
) -> Dict[str, Any]:
    """Answer questions using intelligent query processing"""
    
    from rag_model.rag_utils import retrieve_s3_vectors
    from rag_model.response_summarizer import summarize_for_social_media
    from rag_model.intelligent_fallback import enhance_response_with_fallback
    
    # Use intelligent processing for retrieval
    documents, retrieval_metadata = enhance_retrieval_with_intelligent_processing(
        question, tenant_id, retrieve_s3_vectors
    )
    
    if not documents:
        return {
            "answer": "I don't have information about that in my knowledge base. Could you try rephrasing your question or being more specific?",
            "sources": [],
            "response_type": response_mode,
            "retrieval_metadata": retrieval_metadata
        }
    
    # Generate response using the best available method
    try:
        from rag_model.advanced_aws_rag import AdvancedLLMGenerator, AdvancedRAGConfig
        
        config = AdvancedRAGConfig()
        generator = AdvancedLLMGenerator(config)
        
        # Extract person filter if needed
        person_filter = None
        question_lower = question.lower()
        if "mahi" in question_lower:
            person_filter = "Mahi"
        elif "nitesh" in question_lower:
            person_filter = "Nitesh"
        
        response_data = generator.generate_response(question, documents, person_filter)
        
        # Apply response mode
        if response_mode == "summary":
            summary = summarize_for_social_media(response_data["answer"])
            return {
                "answer": summary,
                "sources": response_data.get("sources", [])[:2],
                "response_type": "summary",
                "retrieval_metadata": retrieval_metadata
            }
        else:
            return {
                "answer": response_data["answer"],
                "sources": response_data.get("sources", []),
                "response_type": "detailed",
                "retrieval_metadata": retrieval_metadata
            }
            
    except Exception as e:
        print(f"Advanced processing failed: {e}")
        
        # Fallback to basic response generation
        context_text = "\n\n".join([doc.page_content for doc in documents])
        basic_answer = f"Based on the available information: {context_text[:1000]}..."
        
        enhanced_answer = enhance_response_with_fallback(question, basic_answer, documents)
        
        return {
            "answer": enhanced_answer,
            "sources": [doc.metadata.get("source", "Unknown") for doc in documents],
            "response_type": response_mode,
            "retrieval_metadata": retrieval_metadata
        }

if __name__ == "__main__":
    # Test the intelligent query processor
    processor = IntelligentQueryProcessor()
    
    test_queries = [
        "What happens on Day 1?",
        "Day 1 process",
        "First day activities",
        "Onboarding procedure"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        result = processor.process_query(query)
        print(f"Original: {result['original_query']}")
        print(f"Enhanced: {result['enhanced_queries']}")
        print(f"Strategy: {result['processing_strategy']}")