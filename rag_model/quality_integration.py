"""
Quality Monitoring Integration for RAG Chatbot
Integrates quality monitoring, performance tracking, and alerting with the existing RAG system.
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from langchain_core.documents import Document

from .quality_monitor import get_quality_monitor, QualityMetrics
from .performance_tracker import get_performance_tracker, PerformanceLog
from .alerting_system import get_alerting_system, AlertingConfig, UserFeedback


class QualityIntegration:
    """
    Integration layer that connects quality monitoring with the RAG system.
    Provides a simple interface for tracking quality and performance metrics.
    """
    
    def __init__(self, log_dir: str = "logs", alerting_config: AlertingConfig = None):
        self.quality_monitor = get_quality_monitor(log_dir=log_dir)
        self.performance_tracker = get_performance_tracker(log_dir=log_dir)
        self.alerting_system = get_alerting_system(config=alerting_config, log_dir=log_dir)
    
    @contextmanager
    def track_request(
        self, 
        tenant_id: int, 
        user_id: str, 
        query: str,
        context_messages: List[Any] = None
    ):
        """
        Context manager for tracking a complete RAG request.
        
        Usage:
            with quality_integration.track_request(tenant_id, user_id, query) as tracker:
                # Retrieval phase
                retrieved_docs = retrieve_documents(query)
                tracker.set_retrieved_docs(retrieved_docs)
                
                # Generation phase
                response = generate_response(query, retrieved_docs)
                tracker.set_response(response)
        """
        tracker = RequestTracker(
            self, tenant_id, user_id, query, context_messages or []
        )
        
        try:
            yield tracker
        finally:
            tracker.finalize()
    
    def record_user_feedback(
        self,
        tenant_id: int,
        user_id: str,
        query: str,
        response: str,
        rating: int,
        feedback_text: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> UserFeedback:
        """Record user feedback and trigger quality analysis"""
        return self.alerting_system.record_user_feedback(
            tenant_id=tenant_id,
            user_id=user_id,
            query=query,
            response=response,
            rating=rating,
            feedback_text=feedback_text,
            session_id=session_id
        )
    
    def get_tenant_quality_summary(self, tenant_id: int, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive quality summary for a tenant"""
        # Get quality metrics summary
        quality_summary = self.quality_monitor.get_tenant_metrics_summary(tenant_id)
        
        # Get performance summary
        performance_summary = self.performance_tracker.get_performance_summary(tenant_id, hours)
        
        # Get feedback summary
        feedback_summary = self.alerting_system.get_feedback_summary(tenant_id, hours)
        
        # Get active alerts
        active_alerts = self.alerting_system.get_active_alerts(tenant_id)
        
        return {
            "tenant_id": tenant_id,
            "time_period": f"Last {hours} hours",
            "quality_metrics": quality_summary,
            "performance_metrics": performance_summary,
            "user_feedback": feedback_summary,
            "active_alerts": [alert.to_dict() for alert in active_alerts],
            "overall_health": self._calculate_overall_health(
                quality_summary, performance_summary, feedback_summary, active_alerts
            ),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_overall_health(
        self, 
        quality_summary: Dict, 
        performance_summary: Dict, 
        feedback_summary: Dict,
        active_alerts: List
    ) -> str:
        """Calculate overall health status"""
        # Check for critical alerts
        if any(alert.severity.value == "critical" for alert in active_alerts):
            return "critical"
        
        # Check performance health
        perf_health = performance_summary.get("health_status", "unknown")
        if perf_health == "critical":
            return "critical"
        elif perf_health == "warning":
            return "warning"
        
        # Check feedback
        if feedback_summary.get("total_feedback", 0) > 0:
            avg_rating = feedback_summary.get("average_rating", 0)
            if avg_rating < 2.0:
                return "critical"
            elif avg_rating < 3.0:
                return "warning"
        
        # Check quality metrics
        if quality_summary.get("total_queries", 0) > 0:
            avg_satisfaction = quality_summary.get("summary", {}).get("avg_user_satisfaction", 0)
            if avg_satisfaction < 0.5:
                return "warning"
            elif avg_satisfaction >= 0.8:
                return "excellent"
        
        return "healthy"


class RequestTracker:
    """
    Tracks metrics for a single RAG request from start to finish.
    """
    
    def __init__(
        self, 
        integration: QualityIntegration, 
        tenant_id: int, 
        user_id: str, 
        query: str,
        context_messages: List[Any]
    ):
        self.integration = integration
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.query = query
        self.context_messages = context_messages
        
        # Timing
        self.start_time = time.time()
        self.retrieval_start_time = None
        self.retrieval_end_time = None
        self.generation_start_time = None
        self.generation_end_time = None
        
        # Data
        self.retrieved_docs: List[Document] = []
        self.response: str = ""
        self.error_occurred: bool = False
        self.error_message: str = ""
    
    def start_retrieval(self):
        """Mark the start of document retrieval"""
        self.retrieval_start_time = time.time()
    
    def end_retrieval(self, retrieved_docs: List[Document]):
        """Mark the end of document retrieval"""
        self.retrieval_end_time = time.time()
        self.retrieved_docs = retrieved_docs
    
    def start_generation(self):
        """Mark the start of response generation"""
        self.generation_start_time = time.time()
    
    def end_generation(self, response: str):
        """Mark the end of response generation"""
        self.generation_end_time = time.time()
        self.response = response
    
    def set_retrieved_docs(self, docs: List[Document]):
        """Set retrieved documents (if not using start/end retrieval)"""
        self.retrieved_docs = docs
    
    def set_response(self, response: str):
        """Set generated response (if not using start/end generation)"""
        self.response = response
    
    def set_error(self, error_message: str):
        """Mark that an error occurred"""
        self.error_occurred = True
        self.error_message = error_message
    
    def finalize(self):
        """Finalize tracking and record all metrics"""
        end_time = time.time()
        total_time_ms = (end_time - self.start_time) * 1000
        
        # Calculate component times
        retrieval_time_ms = 0.0
        if self.retrieval_start_time and self.retrieval_end_time:
            retrieval_time_ms = (self.retrieval_end_time - self.retrieval_start_time) * 1000
        
        generation_time_ms = 0.0
        if self.generation_start_time and self.generation_end_time:
            generation_time_ms = (self.generation_end_time - self.generation_start_time) * 1000
        
        # If component times weren't tracked, estimate them
        if retrieval_time_ms == 0.0 and generation_time_ms == 0.0:
            # Rough estimation: 30% retrieval, 70% generation
            retrieval_time_ms = total_time_ms * 0.3
            generation_time_ms = total_time_ms * 0.7
        elif retrieval_time_ms == 0.0:
            retrieval_time_ms = total_time_ms - generation_time_ms
        elif generation_time_ms == 0.0:
            generation_time_ms = total_time_ms - retrieval_time_ms
        
        # Log performance metrics
        self.integration.performance_tracker.log_performance(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            query=self.query,
            response_time_ms=total_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            num_documents_retrieved=len(self.retrieved_docs),
            num_documents_used=self._count_used_documents(),
            context_length=self._calculate_context_length(),
            response_length=len(self.response),
            error_occurred=self.error_occurred,
            error_message=self.error_message
        )
        
        # Track quality metrics (only if no error occurred)
        if not self.error_occurred and self.response:
            quality_metrics = self.integration.quality_monitor.track_response_quality(
                query=self.query,
                retrieved_docs=self.retrieved_docs,
                response=self.response,
                context_messages=self.context_messages,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                processing_time_ms=total_time_ms
            )
            
            # Check quality thresholds
            self.integration.alerting_system.check_quality_thresholds(quality_metrics)
    
    def _count_used_documents(self) -> int:
        """Estimate how many documents were actually used in the response"""
        if not self.retrieved_docs or not self.response:
            return 0
        
        response_lower = self.response.lower()
        used_count = 0
        
        for doc in self.retrieved_docs:
            # Simple heuristic: check if key phrases from document appear in response
            doc_words = set(doc.page_content.lower().split())
            if len(doc_words) > 10:  # Only check substantial documents
                # Look for 3+ word phrases that appear in both
                doc_phrases = self._extract_phrases(doc.page_content.lower(), 3)
                response_phrases = self._extract_phrases(response_lower, 3)
                
                if doc_phrases.intersection(response_phrases):
                    used_count += 1
        
        return used_count
    
    def _extract_phrases(self, text: str, min_length: int) -> set:
        """Extract phrases of minimum length from text"""
        words = text.split()
        phrases = set()
        
        for i in range(len(words) - min_length + 1):
            phrase = " ".join(words[i:i + min_length])
            phrases.add(phrase)
        
        return phrases
    
    def _calculate_context_length(self) -> int:
        """Calculate total context length"""
        total_length = 0
        
        for msg in self.context_messages:
            if hasattr(msg, 'content'):
                total_length += len(msg.content)
            elif isinstance(msg, dict) and 'content' in msg:
                total_length += len(msg['content'])
            elif isinstance(msg, str):
                total_length += len(msg)
        
        return total_length


# Global integration instance
_quality_integration = None


def get_quality_integration(log_dir: str = "logs", alerting_config: AlertingConfig = None) -> QualityIntegration:
    """
    Get or create the global quality integration instance.
    
    Args:
        log_dir: Directory for log files
        alerting_config: Alerting configuration
        
    Returns:
        QualityIntegration instance
    """
    global _quality_integration
    
    if _quality_integration is None:
        _quality_integration = QualityIntegration(
            log_dir=log_dir, 
            alerting_config=alerting_config
        )
    
    return _quality_integration


def reset_quality_integration():
    """Reset the global quality integration instance (mainly for testing)"""
    global _quality_integration
    _quality_integration = None