"""
Quality Monitoring System for RAG Chatbot
Implements quality metrics calculation, performance logging, and alerting capabilities.
"""

import time
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
import threading
import os

from langchain_core.documents import Document


@dataclass
class QualityMetrics:
    """Quality metrics for a single RAG response"""
    retrieval_confidence: float
    response_relevance: float
    context_utilization: float
    source_diversity: float
    response_completeness: float
    user_satisfaction_prediction: float
    timestamp: datetime = field(default_factory=datetime.now)
    tenant_id: int = 0
    user_id: str = ""
    query: str = ""
    response_length: int = 0
    num_sources: int = 0
    processing_time_ms: float = 0.0


@dataclass
class PerformanceAnalysis:
    """Performance analysis over a time window"""
    tenant_id: int
    time_window_start: datetime
    time_window_end: datetime
    total_queries: int
    avg_retrieval_confidence: float
    avg_response_relevance: float
    avg_context_utilization: float
    avg_source_diversity: float
    avg_response_completeness: float
    avg_user_satisfaction: float
    avg_processing_time_ms: float
    quality_threshold_violations: int
    performance_trend: str  # "improving", "stable", "declining"


@dataclass
class AlertConfig:
    """Configuration for quality monitoring alerts"""
    min_retrieval_confidence: float = 0.5
    min_response_relevance: float = 0.6
    min_context_utilization: float = 0.4
    max_processing_time_ms: float = 5000.0
    violation_threshold: int = 5  # Number of violations before alert
    alert_window_minutes: int = 60


class QualityMonitor:
    """
    Quality monitoring system that tracks and analyzes response quality metrics
    for continuous improvement of the RAG system.
    """
    
    def __init__(self, log_dir: str = "logs", alert_config: AlertConfig = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.alert_config = alert_config or AlertConfig()
        self.metrics_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.violation_counts: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration for quality monitoring"""
        log_file = self.log_dir / "quality_metrics.log"
        
        # Create logger
        self.logger = logging.getLogger("quality_monitor")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def calculate_retrieval_confidence(self, retrieved_docs: List[Document]) -> float:
        """
        Calculate retrieval confidence based on similarity scores and document quality.
        
        Args:
            retrieved_docs: List of retrieved documents with metadata
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not retrieved_docs:
            return 0.0
        
        # Extract similarity scores from metadata
        similarity_scores = []
        for doc in retrieved_docs:
            score = doc.metadata.get("similarity_score", 0.0)
            if isinstance(score, (int, float)):
                similarity_scores.append(float(score))
        
        if not similarity_scores:
            # If no similarity scores available, use content length as proxy
            content_lengths = [len(doc.page_content) for doc in retrieved_docs]
            avg_length = statistics.mean(content_lengths)
            # Normalize based on expected chunk size (1000 chars)
            return min(avg_length / 1000.0, 1.0)
        
        # Calculate confidence based on similarity scores
        avg_similarity = statistics.mean(similarity_scores)
        max_similarity = max(similarity_scores)
        
        # Weight average and max similarity
        confidence = (0.7 * avg_similarity) + (0.3 * max_similarity)
        
        # Boost confidence if we have consistent high scores
        if len(similarity_scores) > 1:
            score_variance = statistics.variance(similarity_scores)
            consistency_bonus = max(0, (0.1 - score_variance) * 2)  # Up to 0.2 bonus
            confidence += consistency_bonus
        
        return min(confidence, 1.0)
    
    def calculate_response_relevance(self, query: str, response: str, retrieved_docs: List[Document]) -> float:
        """
        Calculate response relevance based on query-response alignment and source usage.
        
        Args:
            query: User query
            response: Generated response
            retrieved_docs: Retrieved documents used for generation
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not response or not query:
            return 0.0
        
        # Basic keyword overlap between query and response
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.0
        
        keyword_overlap = len(query_words.intersection(response_words)) / len(query_words)
        
        # Check if response uses information from retrieved documents
        source_usage = 0.0
        if retrieved_docs:
            response_lower = response.lower()
            used_sources = 0
            
            for doc in retrieved_docs:
                # Check if key phrases from document appear in response
                doc_words = set(doc.page_content.lower().split())
                # Look for 3+ word phrases that appear in both
                doc_phrases = self._extract_phrases(doc.page_content.lower(), min_length=3)
                response_phrases = self._extract_phrases(response_lower, min_length=3)
                
                if doc_phrases.intersection(response_phrases):
                    used_sources += 1
            
            source_usage = used_sources / len(retrieved_docs) if retrieved_docs else 0.0
        
        # Response length appropriateness (not too short, not too verbose)
        length_score = self._calculate_length_appropriateness(query, response)
        
        # Combine metrics
        relevance = (0.4 * keyword_overlap) + (0.4 * source_usage) + (0.2 * length_score)
        
        return min(relevance, 1.0)
    
    def calculate_context_utilization(self, context_messages: List[Any], response: str) -> float:
        """
        Calculate how well the response utilizes conversation context.
        
        Args:
            context_messages: Previous conversation messages
            response: Generated response
            
        Returns:
            Context utilization score between 0.0 and 1.0
        """
        if not context_messages or not response:
            return 0.5  # Neutral score when no context available
        
        # Extract text from context messages
        context_text = ""
        for msg in context_messages:
            if hasattr(msg, 'content'):
                context_text += msg.content + " "
            elif isinstance(msg, dict) and 'content' in msg:
                context_text += msg['content'] + " "
            elif isinstance(msg, str):
                context_text += msg + " "
        
        if not context_text.strip():
            return 0.5
        
        # Check for contextual references in response
        contextual_indicators = [
            "as mentioned", "previously", "earlier", "before", "following up",
            "continuing", "regarding your", "about that", "in relation to"
        ]
        
        response_lower = response.lower()
        context_references = sum(1 for indicator in contextual_indicators 
                               if indicator in response_lower)
        
        # Check for topic continuity
        context_words = set(context_text.lower().split())
        response_words = set(response_lower.split())
        
        if context_words:
            topic_continuity = len(context_words.intersection(response_words)) / len(context_words)
        else:
            topic_continuity = 0.0
        
        # Combine metrics
        utilization = min((context_references * 0.2) + (topic_continuity * 0.8), 1.0)
        
        return utilization
    
    def calculate_source_diversity(self, retrieved_docs: List[Document]) -> float:
        """
        Calculate diversity of sources used in retrieval.
        
        Args:
            retrieved_docs: Retrieved documents
            
        Returns:
            Source diversity score between 0.0 and 1.0
        """
        if not retrieved_docs:
            return 0.0
        
        # Extract unique sources
        sources = set()
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "unknown")
            # Normalize source (remove file paths, keep domain/filename)
            if source.startswith("s3://"):
                source = source.split("/")[-1]  # Get filename
            elif source.startswith("http"):
                from urllib.parse import urlparse
                source = urlparse(source).netloc  # Get domain
            sources.add(source)
        
        # Calculate diversity ratio
        diversity = len(sources) / len(retrieved_docs)
        
        # Bonus for having multiple different source types
        source_types = set()
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "")
            if "pdf" in source.lower():
                source_types.add("pdf")
            elif any(domain in source.lower() for domain in ["http", "www", ".com", ".org"]):
                source_types.add("web")
            elif "csv" in source.lower():
                source_types.add("csv")
            else:
                source_types.add("other")
        
        type_diversity_bonus = min(len(source_types) * 0.1, 0.3)
        
        return min(diversity + type_diversity_bonus, 1.0)
    
    def calculate_response_completeness(self, query: str, response: str) -> float:
        """
        Calculate how complete the response is relative to the query.
        
        Args:
            query: User query
            response: Generated response
            
        Returns:
            Completeness score between 0.0 and 1.0
        """
        if not response or not query:
            return 0.0
        
        # Check for question words and ensure they're addressed
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        query_lower = query.lower()
        response_lower = response.lower()
        
        question_coverage = 0.0
        question_count = 0
        
        for word in question_words:
            if word in query_lower:
                question_count += 1
                # Simple heuristic: if response is substantial and contains related terms
                if len(response) > 50 and (
                    word in response_lower or 
                    any(related in response_lower for related in self._get_related_words(word))
                ):
                    question_coverage += 1
        
        if question_count > 0:
            question_score = question_coverage / question_count
        else:
            question_score = 0.8  # Default for non-question queries
        
        # Length appropriateness
        length_score = self._calculate_length_appropriateness(query, response)
        
        # Check for hedging language that might indicate incomplete answers
        hedging_phrases = ["i'm not sure", "might be", "possibly", "perhaps", "i think"]
        hedging_penalty = sum(0.1 for phrase in hedging_phrases if phrase in response_lower)
        
        completeness = (0.6 * question_score) + (0.4 * length_score) - hedging_penalty
        
        return max(min(completeness, 1.0), 0.0)
    
    def predict_user_satisfaction(self, metrics: QualityMetrics) -> float:
        """
        Predict user satisfaction based on other quality metrics.
        
        Args:
            metrics: Calculated quality metrics
            
        Returns:
            Predicted satisfaction score between 0.0 and 1.0
        """
        # Weighted combination of metrics that correlate with user satisfaction
        satisfaction = (
            0.25 * metrics.retrieval_confidence +
            0.30 * metrics.response_relevance +
            0.15 * metrics.context_utilization +
            0.10 * metrics.source_diversity +
            0.20 * metrics.response_completeness
        )
        
        # Apply processing time penalty for slow responses
        if metrics.processing_time_ms > 3000:  # 3 seconds
            time_penalty = min((metrics.processing_time_ms - 3000) / 10000, 0.2)
            satisfaction -= time_penalty
        
        return max(min(satisfaction, 1.0), 0.0)
    
    def track_response_quality(
        self, 
        query: str, 
        retrieved_docs: List[Document],
        response: str, 
        context_messages: List[Any],
        tenant_id: int,
        user_id: str = "unknown",
        processing_time_ms: float = 0.0
    ) -> QualityMetrics:
        """
        Track and calculate quality metrics for a RAG response.
        
        Args:
            query: User query
            retrieved_docs: Documents retrieved for the query
            response: Generated response
            context_messages: Conversation context
            tenant_id: Tenant identifier
            user_id: User identifier
            processing_time_ms: Response processing time in milliseconds
            
        Returns:
            Calculated quality metrics
        """
        start_time = time.time()
        
        # Calculate individual metrics
        retrieval_confidence = self.calculate_retrieval_confidence(retrieved_docs)
        response_relevance = self.calculate_response_relevance(query, response, retrieved_docs)
        context_utilization = self.calculate_context_utilization(context_messages, response)
        source_diversity = self.calculate_source_diversity(retrieved_docs)
        response_completeness = self.calculate_response_completeness(query, response)
        
        # Create metrics object
        metrics = QualityMetrics(
            retrieval_confidence=retrieval_confidence,
            response_relevance=response_relevance,
            context_utilization=context_utilization,
            source_diversity=source_diversity,
            response_completeness=response_completeness,
            user_satisfaction_prediction=0.0,  # Will be calculated next
            tenant_id=tenant_id,
            user_id=user_id,
            query=query[:200],  # Truncate for storage
            response_length=len(response),
            num_sources=len(retrieved_docs),
            processing_time_ms=processing_time_ms
        )
        
        # Calculate predicted satisfaction
        metrics.user_satisfaction_prediction = self.predict_user_satisfaction(metrics)
        
        # Store metrics
        with self.lock:
            self.metrics_history[tenant_id].append(metrics)
        
        # Log metrics
        self._log_metrics(metrics)
        
        # Check for quality threshold violations
        self._check_quality_thresholds(metrics)
        
        calculation_time = (time.time() - start_time) * 1000
        self.logger.debug(f"Quality metrics calculated in {calculation_time:.2f}ms")
        
        return metrics
    
    def _extract_phrases(self, text: str, min_length: int = 3) -> set:
        """Extract phrases of minimum length from text"""
        words = text.split()
        phrases = set()
        
        for i in range(len(words) - min_length + 1):
            phrase = " ".join(words[i:i + min_length])
            phrases.add(phrase)
        
        return phrases
    
    def _get_related_words(self, question_word: str) -> List[str]:
        """Get related words for question words"""
        related = {
            "what": ["definition", "meaning", "description", "explanation"],
            "how": ["method", "process", "way", "steps", "procedure"],
            "why": ["reason", "because", "cause", "purpose"],
            "when": ["time", "date", "period", "schedule"],
            "where": ["location", "place", "position", "site"],
            "who": ["person", "people", "individual", "team"],
            "which": ["option", "choice", "alternative", "selection"]
        }
        return related.get(question_word, [])
    
    def _calculate_length_appropriateness(self, query: str, response: str) -> float:
        """Calculate if response length is appropriate for the query"""
        if not response:
            return 0.0
        
        query_length = len(query.split())
        response_length = len(response.split())
        
        # Expected response length based on query complexity
        if query_length <= 5:
            expected_min, expected_max = 20, 100
        elif query_length <= 15:
            expected_min, expected_max = 50, 200
        else:
            expected_min, expected_max = 100, 300
        
        if expected_min <= response_length <= expected_max:
            return 1.0
        elif response_length < expected_min:
            return response_length / expected_min
        else:
            # Penalty for overly long responses
            excess = response_length - expected_max
            penalty = min(excess / expected_max, 0.5)
            return max(1.0 - penalty, 0.5)
    
    def _log_metrics(self, metrics: QualityMetrics):
        """Log quality metrics to file"""
        log_data = {
            "timestamp": metrics.timestamp.isoformat(),
            "tenant_id": metrics.tenant_id,
            "user_id": metrics.user_id,
            "metrics": {
                "retrieval_confidence": metrics.retrieval_confidence,
                "response_relevance": metrics.response_relevance,
                "context_utilization": metrics.context_utilization,
                "source_diversity": metrics.source_diversity,
                "response_completeness": metrics.response_completeness,
                "user_satisfaction_prediction": metrics.user_satisfaction_prediction,
                "processing_time_ms": metrics.processing_time_ms
            },
            "query_length": len(metrics.query),
            "response_length": metrics.response_length,
            "num_sources": metrics.num_sources
        }
        
        self.logger.info(f"QUALITY_METRICS: {json.dumps(log_data)}")
    
    def _check_quality_thresholds(self, metrics: QualityMetrics):
        """Check if metrics violate quality thresholds and track violations"""
        violations = []
        
        if metrics.retrieval_confidence < self.alert_config.min_retrieval_confidence:
            violations.append(f"Low retrieval confidence: {metrics.retrieval_confidence:.3f}")
        
        if metrics.response_relevance < self.alert_config.min_response_relevance:
            violations.append(f"Low response relevance: {metrics.response_relevance:.3f}")
        
        if metrics.context_utilization < self.alert_config.min_context_utilization:
            violations.append(f"Low context utilization: {metrics.context_utilization:.3f}")
        
        if metrics.processing_time_ms > self.alert_config.max_processing_time_ms:
            violations.append(f"High processing time: {metrics.processing_time_ms:.1f}ms")
        
        if violations:
            with self.lock:
                self.violation_counts[metrics.tenant_id].append({
                    "timestamp": metrics.timestamp,
                    "violations": violations,
                    "query": metrics.query
                })
            
            self.logger.warning(
                f"Quality threshold violations for tenant {metrics.tenant_id}: {violations}"
            )
            
            # Check if we should trigger an alert
            self._check_alert_conditions(metrics.tenant_id)
    
    def _check_alert_conditions(self, tenant_id: int):
        """Check if alert conditions are met for a tenant"""
        with self.lock:
            violations = self.violation_counts[tenant_id]
            
            if len(violations) < self.alert_config.violation_threshold:
                return
            
            # Check if violations occurred within the alert window
            now = datetime.now()
            window_start = now - timedelta(minutes=self.alert_config.alert_window_minutes)
            
            recent_violations = [
                v for v in violations 
                if v["timestamp"] >= window_start
            ]
            
            if len(recent_violations) >= self.alert_config.violation_threshold:
                self._trigger_alert(tenant_id, recent_violations)
    
    def _trigger_alert(self, tenant_id: int, violations: List[Dict]):
        """Trigger quality alert for a tenant"""
        alert_data = {
            "tenant_id": tenant_id,
            "alert_type": "quality_threshold_violation",
            "timestamp": datetime.now().isoformat(),
            "violation_count": len(violations),
            "violations": violations[-5:]  # Last 5 violations
        }
        
        self.logger.error(f"QUALITY_ALERT: {json.dumps(alert_data)}")
        
        # Here you could integrate with external alerting systems
        # (email, Slack, PagerDuty, etc.)
        print(f"🚨 QUALITY ALERT for tenant {tenant_id}: {len(violations)} violations detected")
    
    def analyze_performance_trends(
        self, 
        tenant_id: int, 
        time_window: timedelta = timedelta(hours=24)
    ) -> PerformanceAnalysis:
        """
        Analyze performance trends for a tenant over a specified time window.
        
        Args:
            tenant_id: Tenant identifier
            time_window: Time window for analysis
            
        Returns:
            Performance analysis results
        """
        end_time = datetime.now()
        start_time = end_time - time_window
        
        with self.lock:
            tenant_metrics = list(self.metrics_history[tenant_id])
        
        # Filter metrics within time window
        window_metrics = [
            m for m in tenant_metrics 
            if start_time <= m.timestamp <= end_time
        ]
        
        if not window_metrics:
            return PerformanceAnalysis(
                tenant_id=tenant_id,
                time_window_start=start_time,
                time_window_end=end_time,
                total_queries=0,
                avg_retrieval_confidence=0.0,
                avg_response_relevance=0.0,
                avg_context_utilization=0.0,
                avg_source_diversity=0.0,
                avg_response_completeness=0.0,
                avg_user_satisfaction=0.0,
                avg_processing_time_ms=0.0,
                quality_threshold_violations=0,
                performance_trend="no_data"
            )
        
        # Calculate averages
        total_queries = len(window_metrics)
        avg_retrieval_confidence = statistics.mean(m.retrieval_confidence for m in window_metrics)
        avg_response_relevance = statistics.mean(m.response_relevance for m in window_metrics)
        avg_context_utilization = statistics.mean(m.context_utilization for m in window_metrics)
        avg_source_diversity = statistics.mean(m.source_diversity for m in window_metrics)
        avg_response_completeness = statistics.mean(m.response_completeness for m in window_metrics)
        avg_user_satisfaction = statistics.mean(m.user_satisfaction_prediction for m in window_metrics)
        avg_processing_time_ms = statistics.mean(m.processing_time_ms for m in window_metrics)
        
        # Count quality threshold violations
        violations = 0
        for m in window_metrics:
            if (m.retrieval_confidence < self.alert_config.min_retrieval_confidence or
                m.response_relevance < self.alert_config.min_response_relevance or
                m.context_utilization < self.alert_config.min_context_utilization or
                m.processing_time_ms > self.alert_config.max_processing_time_ms):
                violations += 1
        
        # Determine performance trend
        performance_trend = self._calculate_performance_trend(window_metrics)
        
        return PerformanceAnalysis(
            tenant_id=tenant_id,
            time_window_start=start_time,
            time_window_end=end_time,
            total_queries=total_queries,
            avg_retrieval_confidence=avg_retrieval_confidence,
            avg_response_relevance=avg_response_relevance,
            avg_context_utilization=avg_context_utilization,
            avg_source_diversity=avg_source_diversity,
            avg_response_completeness=avg_response_completeness,
            avg_user_satisfaction=avg_user_satisfaction,
            avg_processing_time_ms=avg_processing_time_ms,
            quality_threshold_violations=violations,
            performance_trend=performance_trend
        )
    
    def _calculate_performance_trend(self, metrics: List[QualityMetrics]) -> str:
        """Calculate performance trend over time"""
        if len(metrics) < 10:
            return "insufficient_data"
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        
        # Split into first and second half
        mid_point = len(sorted_metrics) // 2
        first_half = sorted_metrics[:mid_point]
        second_half = sorted_metrics[mid_point:]
        
        # Calculate average satisfaction for each half
        first_avg = statistics.mean(m.user_satisfaction_prediction for m in first_half)
        second_avg = statistics.mean(m.user_satisfaction_prediction for m in second_half)
        
        # Determine trend
        diff = second_avg - first_avg
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"
    
    def get_tenant_metrics_summary(self, tenant_id: int, limit: int = 100) -> Dict[str, Any]:
        """
        Get summary of recent metrics for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            limit: Maximum number of recent metrics to include
            
        Returns:
            Dictionary containing metrics summary
        """
        with self.lock:
            tenant_metrics = list(self.metrics_history[tenant_id])[-limit:]
        
        if not tenant_metrics:
            return {
                "tenant_id": tenant_id,
                "total_queries": 0,
                "recent_metrics": [],
                "summary": {}
            }
        
        # Calculate summary statistics
        summary = {
            "avg_retrieval_confidence": statistics.mean(m.retrieval_confidence for m in tenant_metrics),
            "avg_response_relevance": statistics.mean(m.response_relevance for m in tenant_metrics),
            "avg_context_utilization": statistics.mean(m.context_utilization for m in tenant_metrics),
            "avg_source_diversity": statistics.mean(m.source_diversity for m in tenant_metrics),
            "avg_response_completeness": statistics.mean(m.response_completeness for m in tenant_metrics),
            "avg_user_satisfaction": statistics.mean(m.user_satisfaction_prediction for m in tenant_metrics),
            "avg_processing_time_ms": statistics.mean(m.processing_time_ms for m in tenant_metrics),
            "total_queries": len(tenant_metrics),
            "date_range": {
                "start": min(m.timestamp for m in tenant_metrics).isoformat(),
                "end": max(m.timestamp for m in tenant_metrics).isoformat()
            }
        }
        
        # Convert metrics to serializable format
        recent_metrics = [asdict(m) for m in tenant_metrics[-20:]]  # Last 20 metrics
        for metric in recent_metrics:
            metric["timestamp"] = metric["timestamp"].isoformat()
        
        return {
            "tenant_id": tenant_id,
            "total_queries": len(tenant_metrics),
            "recent_metrics": recent_metrics,
            "summary": summary
        }
    
    def export_metrics_to_file(self, tenant_id: int, filepath: str):
        """
        Export tenant metrics to a JSON file.
        
        Args:
            tenant_id: Tenant identifier
            filepath: Output file path
        """
        summary = self.get_tenant_metrics_summary(tenant_id, limit=1000)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Exported metrics for tenant {tenant_id} to {filepath}")
    
    def clear_tenant_metrics(self, tenant_id: int):
        """Clear stored metrics for a tenant"""
        with self.lock:
            if tenant_id in self.metrics_history:
                self.metrics_history[tenant_id].clear()
            if tenant_id in self.violation_counts:
                self.violation_counts[tenant_id].clear()
        
        self.logger.info(f"Cleared metrics for tenant {tenant_id}")
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """
        Get overall system health status across all tenants.
        
        Returns:
            System health status dictionary
        """
        with self.lock:
            all_metrics = []
            tenant_count = len(self.metrics_history)
            
            for tenant_metrics in self.metrics_history.values():
                all_metrics.extend(list(tenant_metrics))
        
        if not all_metrics:
            return {
                "status": "no_data",
                "tenant_count": 0,
                "total_queries": 0,
                "system_averages": {}
            }
        
        # Calculate system-wide averages
        system_averages = {
            "retrieval_confidence": statistics.mean(m.retrieval_confidence for m in all_metrics),
            "response_relevance": statistics.mean(m.response_relevance for m in all_metrics),
            "context_utilization": statistics.mean(m.context_utilization for m in all_metrics),
            "source_diversity": statistics.mean(m.source_diversity for m in all_metrics),
            "response_completeness": statistics.mean(m.response_completeness for m in all_metrics),
            "user_satisfaction": statistics.mean(m.user_satisfaction_prediction for m in all_metrics),
            "processing_time_ms": statistics.mean(m.processing_time_ms for m in all_metrics)
        }
        
        # Determine overall health status
        avg_satisfaction = system_averages["user_satisfaction"]
        if avg_satisfaction >= 0.8:
            status = "excellent"
        elif avg_satisfaction >= 0.7:
            status = "good"
        elif avg_satisfaction >= 0.6:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "status": status,
            "tenant_count": tenant_count,
            "total_queries": len(all_metrics),
            "system_averages": system_averages,
            "last_updated": datetime.now().isoformat()
        }


# Global quality monitor instance
_quality_monitor = None
_monitor_lock = threading.Lock()


def get_quality_monitor(log_dir: str = "logs", alert_config: AlertConfig = None) -> QualityMonitor:
    """
    Get or create the global quality monitor instance.
    
    Args:
        log_dir: Directory for log files
        alert_config: Alert configuration
        
    Returns:
        QualityMonitor instance
    """
    global _quality_monitor
    
    with _monitor_lock:
        if _quality_monitor is None:
            _quality_monitor = QualityMonitor(log_dir=log_dir, alert_config=alert_config)
        
        return _quality_monitor


def reset_quality_monitor():
    """Reset the global quality monitor instance (mainly for testing)"""
    global _quality_monitor
    
    with _monitor_lock:
        _quality_monitor = None