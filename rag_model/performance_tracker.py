"""
Performance Logging and Tracking System for RAG Chatbot
Implements comprehensive logging of retrieval performance and metrics aggregation.
"""

import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
import statistics
import os
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import csv

from .quality_monitor import QualityMetrics


@dataclass
class PerformanceLog:
    """Individual performance log entry"""
    timestamp: datetime
    tenant_id: int
    user_id: str
    query: str
    response_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    num_documents_retrieved: int
    num_documents_used: int
    context_length: int
    response_length: int
    error_occurred: bool = False
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics over a time period"""
    tenant_id: int
    time_period_start: datetime
    time_period_end: datetime
    total_requests: int
    successful_requests: int
    error_rate: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    avg_retrieval_time_ms: float
    avg_generation_time_ms: float
    avg_documents_retrieved: float
    avg_documents_used: float
    avg_context_length: float
    avg_response_length: float
    requests_per_minute: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['time_period_start'] = self.time_period_start.isoformat()
        data['time_period_end'] = self.time_period_end.isoformat()
        return data


class PerformanceTracker:
    """
    Performance tracking system that logs detailed performance metrics
    and provides aggregation and trend analysis capabilities.
    """
    
    def __init__(self, log_dir: str = "logs", db_path: str = None, max_memory_logs: int = 10000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.db_path = db_path or str(self.log_dir / "performance.db")
        self.max_memory_logs = max_memory_logs
        
        # In-memory storage for recent logs
        self.recent_logs: deque = deque(maxlen=max_memory_logs)
        self.tenant_logs: Dict[int, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="perf_tracker")
        
        # Setup logging and database
        self._setup_logging()
        self._setup_database()
        
        # Aggregation cache
        self._aggregation_cache: Dict[str, Tuple[datetime, AggregatedMetrics]] = {}
        self._cache_ttl = timedelta(minutes=5)
    
    def _setup_logging(self):
        """Setup performance logging configuration"""
        # Performance log file
        perf_log_file = self.log_dir / "performance.log"
        
        # Create performance logger
        self.perf_logger = logging.getLogger("performance_tracker")
        self.perf_logger.setLevel(logging.INFO)
        
        # Create file handler with rotation
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            perf_log_file, 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.perf_logger.handlers:
            self.perf_logger.addHandler(handler)
        
        # Error log file
        error_log_file = self.log_dir / "performance_errors.log"
        
        # Create error logger
        self.error_logger = logging.getLogger("performance_errors")
        self.error_logger.setLevel(logging.ERROR)
        
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        if not self.error_logger.handlers:
            self.error_logger.addHandler(error_handler)
    
    def _setup_database(self):
        """Setup SQLite database for persistent storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        tenant_id INTEGER NOT NULL,
                        user_id TEXT NOT NULL,
                        query TEXT NOT NULL,
                        response_time_ms REAL NOT NULL,
                        retrieval_time_ms REAL NOT NULL,
                        generation_time_ms REAL NOT NULL,
                        num_documents_retrieved INTEGER NOT NULL,
                        num_documents_used INTEGER NOT NULL,
                        context_length INTEGER NOT NULL,
                        response_length INTEGER NOT NULL,
                        error_occurred BOOLEAN NOT NULL,
                        error_message TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_tenant_timestamp 
                    ON performance_logs(tenant_id, timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON performance_logs(timestamp)
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS aggregated_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tenant_id INTEGER NOT NULL,
                        time_period_start TEXT NOT NULL,
                        time_period_end TEXT NOT NULL,
                        aggregation_type TEXT NOT NULL,
                        metrics_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_aggregated_tenant_period 
                    ON aggregated_metrics(tenant_id, time_period_start, aggregation_type)
                """)
                
                conn.commit()
                
        except Exception as e:
            self.error_logger.error(f"Failed to setup database: {e}")
            raise
    
    def log_performance(
        self,
        tenant_id: int,
        user_id: str,
        query: str,
        response_time_ms: float,
        retrieval_time_ms: float,
        generation_time_ms: float,
        num_documents_retrieved: int,
        num_documents_used: int,
        context_length: int,
        response_length: int,
        error_occurred: bool = False,
        error_message: str = ""
    ) -> PerformanceLog:
        """
        Log performance metrics for a single request.
        
        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            query: User query (truncated for storage)
            response_time_ms: Total response time in milliseconds
            retrieval_time_ms: Document retrieval time in milliseconds
            generation_time_ms: Response generation time in milliseconds
            num_documents_retrieved: Number of documents retrieved
            num_documents_used: Number of documents actually used in response
            context_length: Length of conversation context
            response_length: Length of generated response
            error_occurred: Whether an error occurred
            error_message: Error message if applicable
            
        Returns:
            PerformanceLog entry
        """
        log_entry = PerformanceLog(
            timestamp=datetime.now(),
            tenant_id=tenant_id,
            user_id=user_id,
            query=query[:200],  # Truncate for storage
            response_time_ms=response_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            num_documents_retrieved=num_documents_retrieved,
            num_documents_used=num_documents_used,
            context_length=context_length,
            response_length=response_length,
            error_occurred=error_occurred,
            error_message=error_message[:500] if error_message else ""
        )
        
        # Store in memory
        with self.lock:
            self.recent_logs.append(log_entry)
            self.tenant_logs[tenant_id].append(log_entry)
        
        # Log to file
        self._log_to_file(log_entry)
        
        # Store in database (async)
        self.executor.submit(self._store_to_database, log_entry)
        
        return log_entry
    
    def _log_to_file(self, log_entry: PerformanceLog):
        """Log performance entry to file"""
        log_data = log_entry.to_dict()
        
        if log_entry.error_occurred:
            self.error_logger.error(f"PERFORMANCE_ERROR: {json.dumps(log_data)}")
        else:
            self.perf_logger.info(f"PERFORMANCE: {json.dumps(log_data)}")
    
    def _store_to_database(self, log_entry: PerformanceLog):
        """Store performance entry to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_logs (
                        timestamp, tenant_id, user_id, query, response_time_ms,
                        retrieval_time_ms, generation_time_ms, num_documents_retrieved,
                        num_documents_used, context_length, response_length,
                        error_occurred, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_entry.timestamp.isoformat(),
                    log_entry.tenant_id,
                    log_entry.user_id,
                    log_entry.query,
                    log_entry.response_time_ms,
                    log_entry.retrieval_time_ms,
                    log_entry.generation_time_ms,
                    log_entry.num_documents_retrieved,
                    log_entry.num_documents_used,
                    log_entry.context_length,
                    log_entry.response_length,
                    log_entry.error_occurred,
                    log_entry.error_message
                ))
                conn.commit()
                
        except Exception as e:
            self.error_logger.error(f"Failed to store performance log to database: {e}")
    
    def get_recent_performance(self, tenant_id: int = None, limit: int = 100) -> List[PerformanceLog]:
        """
        Get recent performance logs.
        
        Args:
            tenant_id: Optional tenant filter
            limit: Maximum number of logs to return
            
        Returns:
            List of recent performance logs
        """
        with self.lock:
            if tenant_id is not None:
                logs = list(self.tenant_logs[tenant_id])
            else:
                logs = list(self.recent_logs)
        
        # Sort by timestamp (most recent first) and limit
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return logs[:limit]
    
    def aggregate_metrics(
        self,
        tenant_id: int,
        start_time: datetime,
        end_time: datetime,
        use_cache: bool = True
    ) -> AggregatedMetrics:
        """
        Aggregate performance metrics for a time period.
        
        Args:
            tenant_id: Tenant identifier
            start_time: Start of time period
            end_time: End of time period
            use_cache: Whether to use cached results
            
        Returns:
            Aggregated metrics
        """
        # Check cache first
        cache_key = f"{tenant_id}_{start_time.isoformat()}_{end_time.isoformat()}"
        
        if use_cache and cache_key in self._aggregation_cache:
            cached_time, cached_metrics = self._aggregation_cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached_metrics
        
        # Get logs from database for the time period
        logs = self._get_logs_from_database(tenant_id, start_time, end_time)
        
        if not logs:
            return AggregatedMetrics(
                tenant_id=tenant_id,
                time_period_start=start_time,
                time_period_end=end_time,
                total_requests=0,
                successful_requests=0,
                error_rate=0.0,
                avg_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                avg_retrieval_time_ms=0.0,
                avg_generation_time_ms=0.0,
                avg_documents_retrieved=0.0,
                avg_documents_used=0.0,
                avg_context_length=0.0,
                avg_response_length=0.0,
                requests_per_minute=0.0
            )
        
        # Calculate metrics
        total_requests = len(logs)
        successful_requests = sum(1 for log in logs if not log.error_occurred)
        error_rate = (total_requests - successful_requests) / total_requests if total_requests > 0 else 0.0
        
        # Response time metrics
        response_times = [log.response_time_ms for log in logs]
        avg_response_time_ms = statistics.mean(response_times)
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        p95_index = int(0.95 * len(sorted_times))
        p99_index = int(0.99 * len(sorted_times))
        p95_response_time_ms = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
        p99_response_time_ms = sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1]
        
        # Other averages
        avg_retrieval_time_ms = statistics.mean(log.retrieval_time_ms for log in logs)
        avg_generation_time_ms = statistics.mean(log.generation_time_ms for log in logs)
        avg_documents_retrieved = statistics.mean(log.num_documents_retrieved for log in logs)
        avg_documents_used = statistics.mean(log.num_documents_used for log in logs)
        avg_context_length = statistics.mean(log.context_length for log in logs)
        avg_response_length = statistics.mean(log.response_length for log in logs)
        
        # Calculate requests per minute
        time_diff_minutes = (end_time - start_time).total_seconds() / 60
        requests_per_minute = total_requests / time_diff_minutes if time_diff_minutes > 0 else 0.0
        
        aggregated = AggregatedMetrics(
            tenant_id=tenant_id,
            time_period_start=start_time,
            time_period_end=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            error_rate=error_rate,
            avg_response_time_ms=avg_response_time_ms,
            p95_response_time_ms=p95_response_time_ms,
            p99_response_time_ms=p99_response_time_ms,
            avg_retrieval_time_ms=avg_retrieval_time_ms,
            avg_generation_time_ms=avg_generation_time_ms,
            avg_documents_retrieved=avg_documents_retrieved,
            avg_documents_used=avg_documents_used,
            avg_context_length=avg_context_length,
            avg_response_length=avg_response_length,
            requests_per_minute=requests_per_minute
        )
        
        # Cache the result
        self._aggregation_cache[cache_key] = (datetime.now(), aggregated)
        
        # Store aggregated metrics in database
        self.executor.submit(self._store_aggregated_metrics, aggregated, "custom")
        
        return aggregated
    
    def _get_logs_from_database(
        self, 
        tenant_id: int, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[PerformanceLog]:
        """Get performance logs from database for a time period"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, tenant_id, user_id, query, response_time_ms,
                           retrieval_time_ms, generation_time_ms, num_documents_retrieved,
                           num_documents_used, context_length, response_length,
                           error_occurred, error_message
                    FROM performance_logs
                    WHERE tenant_id = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """, (tenant_id, start_time.isoformat(), end_time.isoformat()))
                
                logs = []
                for row in cursor.fetchall():
                    log = PerformanceLog(
                        timestamp=datetime.fromisoformat(row[0]),
                        tenant_id=row[1],
                        user_id=row[2],
                        query=row[3],
                        response_time_ms=row[4],
                        retrieval_time_ms=row[5],
                        generation_time_ms=row[6],
                        num_documents_retrieved=row[7],
                        num_documents_used=row[8],
                        context_length=row[9],
                        response_length=row[10],
                        error_occurred=bool(row[11]),
                        error_message=row[12] or ""
                    )
                    logs.append(log)
                
                return logs
                
        except Exception as e:
            self.error_logger.error(f"Failed to get logs from database: {e}")
            return []
    
    def _store_aggregated_metrics(self, metrics: AggregatedMetrics, aggregation_type: str):
        """Store aggregated metrics to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO aggregated_metrics (
                        tenant_id, time_period_start, time_period_end,
                        aggregation_type, metrics_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metrics.tenant_id,
                    metrics.time_period_start.isoformat(),
                    metrics.time_period_end.isoformat(),
                    aggregation_type,
                    json.dumps(metrics.to_dict()),
                    datetime.now().isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            self.error_logger.error(f"Failed to store aggregated metrics: {e}")
    
    def generate_hourly_aggregations(self, tenant_id: int, date: datetime.date = None):
        """Generate hourly aggregations for a specific date"""
        if date is None:
            date = datetime.now().date()
        
        start_of_day = datetime.combine(date, datetime.min.time())
        
        for hour in range(24):
            hour_start = start_of_day + timedelta(hours=hour)
            hour_end = hour_start + timedelta(hours=1)
            
            metrics = self.aggregate_metrics(tenant_id, hour_start, hour_end, use_cache=False)
            self._store_aggregated_metrics(metrics, "hourly")
    
    def generate_daily_aggregations(self, tenant_id: int, date: datetime.date = None):
        """Generate daily aggregations for a specific date"""
        if date is None:
            date = datetime.now().date()
        
        day_start = datetime.combine(date, datetime.min.time())
        day_end = day_start + timedelta(days=1)
        
        metrics = self.aggregate_metrics(tenant_id, day_start, day_end, use_cache=False)
        self._store_aggregated_metrics(metrics, "daily")
    
    def export_performance_data(
        self, 
        tenant_id: int, 
        start_time: datetime, 
        end_time: datetime,
        output_file: str,
        format: str = "csv"
    ):
        """
        Export performance data to file.
        
        Args:
            tenant_id: Tenant identifier
            start_time: Start time for export
            end_time: End time for export
            output_file: Output file path
            format: Export format ("csv" or "json")
        """
        logs = self._get_logs_from_database(tenant_id, start_time, end_time)
        
        if format.lower() == "csv":
            self._export_to_csv(logs, output_file)
        elif format.lower() == "json":
            self._export_to_json(logs, output_file)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_to_csv(self, logs: List[PerformanceLog], output_file: str):
        """Export logs to CSV file"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            if not logs:
                return
            
            fieldnames = list(logs[0].to_dict().keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for log in logs:
                writer.writerow(log.to_dict())
    
    def _export_to_json(self, logs: List[PerformanceLog], output_file: str):
        """Export logs to JSON file"""
        data = [log.to_dict() for log in logs]
        
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=2, default=str)
    
    def get_performance_summary(self, tenant_id: int, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for the last N hours.
        
        Args:
            tenant_id: Tenant identifier
            hours: Number of hours to look back
            
        Returns:
            Performance summary dictionary
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        metrics = self.aggregate_metrics(tenant_id, start_time, end_time)
        
        # Get recent error logs
        recent_logs = self.get_recent_performance(tenant_id, limit=100)
        recent_errors = [log for log in recent_logs if log.error_occurred]
        
        return {
            "tenant_id": tenant_id,
            "time_period": f"Last {hours} hours",
            "aggregated_metrics": metrics.to_dict(),
            "recent_errors": [log.to_dict() for log in recent_errors[:10]],
            "health_status": self._calculate_health_status(metrics),
            "generated_at": datetime.now().isoformat()
        }
    
    def _calculate_health_status(self, metrics: AggregatedMetrics) -> str:
        """Calculate health status based on metrics"""
        if metrics.total_requests == 0:
            return "no_data"
        
        # Check error rate
        if metrics.error_rate > 0.1:  # More than 10% errors
            return "critical"
        elif metrics.error_rate > 0.05:  # More than 5% errors
            return "warning"
        
        # Check response times
        if metrics.avg_response_time_ms > 5000:  # More than 5 seconds
            return "warning"
        elif metrics.p95_response_time_ms > 10000:  # 95th percentile > 10 seconds
            return "warning"
        
        return "healthy"
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old performance data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete old performance logs
                cursor = conn.execute("""
                    DELETE FROM performance_logs 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_logs = cursor.rowcount
                
                # Delete old aggregated metrics
                cursor = conn.execute("""
                    DELETE FROM aggregated_metrics 
                    WHERE time_period_end < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_aggregations = cursor.rowcount
                
                conn.commit()
                
                self.perf_logger.info(
                    f"Cleaned up {deleted_logs} performance logs and "
                    f"{deleted_aggregations} aggregated metrics older than {days_to_keep} days"
                )
                
        except Exception as e:
            self.error_logger.error(f"Failed to cleanup old data: {e}")
    
    def shutdown(self):
        """Shutdown the performance tracker"""
        self.executor.shutdown(wait=True)


# Global performance tracker instance
_performance_tracker = None
_tracker_lock = threading.Lock()


def get_performance_tracker(log_dir: str = "logs", db_path: str = None) -> PerformanceTracker:
    """
    Get or create the global performance tracker instance.
    
    Args:
        log_dir: Directory for log files
        db_path: Path to SQLite database file
        
    Returns:
        PerformanceTracker instance
    """
    global _performance_tracker
    
    with _tracker_lock:
        if _performance_tracker is None:
            _performance_tracker = PerformanceTracker(log_dir=log_dir, db_path=db_path)
        
        return _performance_tracker


def reset_performance_tracker():
    """Reset the global performance tracker instance (mainly for testing)"""
    global _performance_tracker
    
    with _tracker_lock:
        if _performance_tracker is not None:
            _performance_tracker.shutdown()
        _performance_tracker = None