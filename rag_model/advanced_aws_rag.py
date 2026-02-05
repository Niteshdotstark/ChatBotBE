#!/usr/bin/env python3
"""
Advanced RAG System using AWS Bedrock's latest models and services
for improved retrieval and chunking accuracy.
"""

import boto3
import json
import os
from typing import List, Dict, Any, Optional
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import numpy as np
from dataclasses import dataclass

@dataclass
class AdvancedRAGConfig:
    """Configuration for advanced RAG system"""
    # Best working models in ap-south-1 region (tested and confirmed)
    llm_model: str = "anthropic.claude-3-sonnet-20240229-v1:0"  # Claude 3 Sonnet (working in ap-south-1)
    embedding_model: str = "amazon.titan-embed-text-v2:0"  # Latest Titan embeddings (confirmed working)
    
    # Advanced chunking parameters
    chunk_size: int = 800  # Smaller chunks for better precision
    chunk_overlap: int = 100
    
    # Retrieval parameters
    top_k: int = 5  # Fewer but more relevant chunks
    similarity_threshold: float = 0.3  # Lower threshold for better recall
    
    # LLM parameters
    temperature: float = 0.05  # Very low for maximum accuracy
    max_tokens: int = 4000
    
    # AWS region - using ap-south-1 as requested
    region_name: str = "ap-south-1"

class AdvancedDocumentProcessor:
    """Advanced document processing with person-aware chunking"""
    
    def __init__(self, config: AdvancedRAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def process_document(self, content: str, source: str) -> List[Document]:
        """Process document with person-aware chunking"""
        
        # Detect person names in the document
        person_names = self._extract_person_names(content)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Determine which person this chunk belongs to
            chunk_person = self._identify_chunk_person(chunk, person_names)
            
            # Create metadata with person information
            metadata = {
                "source": source,
                "chunk_id": i,
                "person": chunk_person,
                "chunk_size": len(chunk)
            }
            
            # Add person prefix to chunk content for better retrieval
            if chunk_person:
                enhanced_content = f"[PERSON: {chunk_person}] {chunk}"
            else:
                enhanced_content = chunk
            
            documents.append(Document(
                page_content=enhanced_content,
                metadata=metadata
            ))
        
        return documents
    
    def _extract_person_names(self, content: str) -> List[str]:
        """Extract person names from document content"""
        content_lower = content.lower()
        names = []
        
        # Common patterns for names in resumes
        if "mahi" in content_lower or "maheshwari" in content_lower:
            names.append("Mahi")
        if "nitesh" in content_lower:
            names.append("Nitesh")
            
        return names
    
    def _identify_chunk_person(self, chunk: str, person_names: List[str]) -> Optional[str]:
        """Identify which person a chunk belongs to"""
        chunk_lower = chunk.lower()
        
        # Direct name mentions
        for name in person_names:
            if name.lower() in chunk_lower:
                return name
        
        # Context-based identification
        # Look for personal indicators
        personal_indicators = [
            "experience", "skills", "projects", "education", 
            "worked", "developed", "built", "created"
        ]
        
        if any(indicator in chunk_lower for indicator in personal_indicators):
            # If chunk has personal content but no name, try to infer from context
            # This is where we could use more sophisticated NLP
            pass
        
        return None

class AdvancedRetriever:
    """Advanced retrieval with semantic filtering and re-ranking"""
    
    def __init__(self, config: AdvancedRAGConfig):
        self.config = config
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=config.region_name)
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id=config.embedding_model,
            region_name=config.region_name
        )
    
    def retrieve_documents(
        self, 
        query: str, 
        tenant_id: int, 
        person_filter: Optional[str] = None
    ) -> List[Document]:
        """Retrieve documents with advanced filtering and re-ranking"""
        
        # Step 1: Get initial candidates using vector similarity
        candidates = self._get_vector_candidates(query, tenant_id)
        
        # Step 2: Apply person filtering
        if person_filter:
            candidates = self._filter_by_person(candidates, person_filter)
        
        # Step 3: Re-rank using semantic similarity
        ranked_docs = self._semantic_rerank(query, candidates)
        
        # Step 4: Apply similarity threshold
        filtered_docs = [
            doc for doc in ranked_docs 
            if doc.metadata.get("similarity_score", 0) >= self.config.similarity_threshold
        ]
        
        return filtered_docs[:self.config.top_k]
    
    def _get_vector_candidates(self, query: str, tenant_id: int) -> List[Document]:
        """Get initial candidates using vector search"""
        # This would integrate with your existing S3 Vectors or OpenSearch
        # For now, using the existing retrieve_s3_vectors function
        from rag_model.rag_utils import retrieve_s3_vectors
        return retrieve_s3_vectors(query, tenant_id, top_k=10)
    
    def _filter_by_person(self, documents: List[Document], person: str) -> List[Document]:
        """Filter documents by person with advanced matching"""
        filtered = []
        person_lower = person.lower()
        
        print(f"🔍 Filtering {len(documents)} documents for person: '{person}'")
        
        for i, doc in enumerate(documents):
            doc_person = doc.metadata.get("person", "").lower()
            source = doc.metadata.get("source", "")
            content_preview = doc.page_content[:100]
            
            print(f"  Doc {i+1}: person='{doc_person}', source='{source.split('/')[-1] if '/' in source else source}'")
            
            # Primary check: metadata person field (most reliable)
            if doc_person == person_lower:
                filtered.append(doc)
                print(f"    ✅ MATCH: metadata person field")
                continue
            
            # Secondary check: person tag in content (for enhanced chunks)
            content = doc.page_content.lower()
            if f"[person: {person_lower}]" in content:
                filtered.append(doc)
                print(f"    ✅ MATCH: person tag in content")
                continue
            
            # Tertiary check: source filename (only if person not already identified)
            if doc_person == "unknown" or doc_person == "":
                source_lower = source.lower()
                if person_lower in source_lower:
                    filtered.append(doc)
                    print(f"    ✅ MATCH: person in filename")
                    continue
            
            print(f"    ❌ NO MATCH")
        
        print(f"🔍 Person filtering result: {len(documents)} → {len(filtered)} documents for '{person}'")
        return filtered
    
    def _semantic_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Re-rank documents using semantic similarity"""
        if not documents:
            return documents
        
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Get document embeddings and calculate similarities
            doc_similarities = []
            for doc in documents:
                try:
                    doc_embedding = self.embeddings.embed_query(doc.page_content)
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    doc.metadata["similarity_score"] = similarity
                    doc_similarities.append((doc, similarity))
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    doc.metadata["similarity_score"] = 0.0
                    doc_similarities.append((doc, 0.0))
            
            # Sort by similarity
            doc_similarities.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in doc_similarities]
            
        except Exception as e:
            print(f"Error in semantic re-ranking: {e}")
            return documents
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

class AdvancedLLMGenerator:
    """Advanced LLM generation with Claude 3.5 Sonnet"""
    
    def __init__(self, config: AdvancedRAGConfig):
        self.config = config
        self.llm = ChatBedrock(
            model_id=config.llm_model,
            region_name=config.region_name,
            model_kwargs={
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": 0.9,
                "top_k": 250
            }
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are a highly accurate information extraction assistant. Your primary goal is to provide precise, factual information based solely on the provided context.

CRITICAL ACCURACY RULES:
1. ONLY use information explicitly stated in the provided context documents
2. When asked about a specific person, ONLY provide information that clearly belongs to that person
3. If information is ambiguous or could belong to multiple people, do not include it
4. If you cannot find specific information about the requested person, clearly state this
5. Never mix or confuse information between different people
6. Pay close attention to document sources and metadata to ensure accuracy

RESPONSE FORMAT:
- Be direct and factual
- Use bullet points for lists
- Include specific details when available
- Cite information confidence level when uncertain

PERSON IDENTIFICATION:
- Look for explicit name mentions in the context
- Check document metadata for person identification
- Verify information belongs to the specific person asked about
"""),
            ("human", """
Context Documents:
{context}

Query: {query}
Person Filter: {person_filter}

Instructions: Provide accurate information about {person_filter} based ONLY on the context documents above. If the context doesn't contain information about {person_filter}, clearly state this.

Response:
""")
        ])
    
    def generate_response(
        self, 
        query: str, 
        documents: List[Document], 
        person_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using advanced LLM"""
        
        # Prepare context from documents (without document numbers)
        context_parts = []
        for doc in documents:
            # Just include the content without document numbering or metadata
            context_parts.append(doc.page_content)
        
        context = "\n\n".join(context_parts)
        
        # Create adaptive prompt based on whether it's person-specific or general
        if person_filter and person_filter.lower() != "none":
            # Person-specific query
            instructions = f"Provide accurate information about {person_filter} based ONLY on the context documents above. If the context doesn't contain information about {person_filter}, clearly state this."
        else:
            # General query
            instructions = "Provide accurate information to answer the query based ONLY on the context documents above. If the context doesn't contain relevant information, clearly state this."
        
        try:
            # Generate response with adaptive prompt
            response = self.llm.invoke([
                SystemMessage(content="""
You are a helpful and knowledgeable assistant. Provide clear, accurate, and direct responses.

RESPONSE GUIDELINES:
1. Answer directly and naturally, as if you're an expert on the topic
2. Don't mention "context", "documents", or "provided information" - just answer the question
3. Be comprehensive but concise
4. Use bullet points or numbered lists when helpful for clarity
5. Write in a confident, professional tone as if this is your knowledge
6. If you don't have enough information to answer fully, say "I don't have complete information about..." instead of referencing context
7. Start answers directly - avoid phrases like "Based on...", "According to...", "The context shows..."

IMPORTANT: Answer as if you're an expert who knows this information, not as someone reading from documents.
"""),
                HumanMessage(content=f"""
Here's the relevant information:
{context}

Question: {query}

Please provide a direct, helpful answer to the question. {instructions}
""")
            ])
            
            return {
                "answer": response.content,
                "sources": [doc.metadata.get("source", "Unknown") for doc in documents],
                "confidence": self._calculate_confidence(documents),
                "person_filter": person_filter
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "person_filter": person_filter
            }
    
    def _calculate_confidence(self, documents: List[Document]) -> float:
        """Calculate confidence score based on document quality"""
        if not documents:
            return 0.0
        
        similarities = [doc.metadata.get("similarity_score", 0.0) for doc in documents]
        return sum(similarities) / len(similarities)

class AdvancedRAGSystem:
    """Complete advanced RAG system"""
    
    def __init__(self, config: Optional[AdvancedRAGConfig] = None):
        self.config = config or AdvancedRAGConfig()
        self.processor = AdvancedDocumentProcessor(self.config)
        self.retriever = AdvancedRetriever(self.config)
        self.generator = AdvancedLLMGenerator(self.config)
    
    def query(
        self, 
        question: str, 
        tenant_id: int, 
        person_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main query interface"""
        
        print(f"🔍 Advanced RAG Query: {question}")
        if person_filter:
            print(f"🎯 Person Filter: {person_filter}")
        
        # Step 1: Retrieve relevant documents
        documents = self.retriever.retrieve_documents(question, tenant_id, person_filter)
        print(f"📄 Retrieved {len(documents)} documents")
        
        # Step 2: Generate response
        response = self.generator.generate_response(question, documents, person_filter)
        print(f"✅ Generated response with confidence: {response['confidence']:.3f}")
        
        return response

# Integration function for existing system
def answer_question_advanced(
    question: str, 
    tenant_id: int, 
    person_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Advanced answer function using latest AWS Bedrock models"""
    
    config = AdvancedRAGConfig()
    rag_system = AdvancedRAGSystem(config)
    
    return rag_system.query(question, tenant_id, person_filter)

if __name__ == "__main__":
    # Test the advanced system
    config = AdvancedRAGConfig()
    print(f"🚀 Advanced RAG System Configuration:")
    print(f"   LLM Model: {config.llm_model}")
    print(f"   Embedding Model: {config.embedding_model}")
    print(f"   Temperature: {config.temperature}")
    print(f"   Chunk Size: {config.chunk_size}")