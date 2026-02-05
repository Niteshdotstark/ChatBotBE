"""
Alerting System for RAG Chatbot Quality Monitoring
Implements threshold-based alerting and feedback correlation with quality metrics.
"""

import json
import time
import logging
import smtplib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
import requests
import os

from .quality_monitor import QualityMetrics, AlertConfig
from .performance_tracker import PerformanceLog, AggregatedMetrics


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts"""
    QUALITY_THRESHOLD = "quality_threshold"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_HIGH = "error_rate_high"
    RESPONSE_TIME_HIGH = "response_time_high"
    USER_FEEDBACK_NEGATIVE = "user_feedback_negative"
    SYSTEM_HEALTH = "system_health"


@dataclass
class UserFeedback:
    """User feedback data structure"""
    tenant_id: int
    user_id: str
    query: str
    response: str
    rating: int  # 1-5 scale
    feedback_text: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    tenant_id: int
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['alert_type'] = self.alert_type.value
        data['severity'] = self.severity.value
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


@dataclass
class AlertingConfig:
    """Configuration for the alerting system"""
    # Quality thresholds
    min_retrieval_confidence: float = 0.5
    min_response_relevance: float = 0.6
    min_context_utilization: float = 0.4
    min_user_satisfaction: float = 0.6
    
    # Performance thresholds
    max_response_time_ms: float = 5000.0
    max_error_rate: float = 0.1
    max_p95_response_time_ms: float = 8000.0
    
    # Alert conditions
    violation_threshold: int = 5  # Number of violations before alert
    alert_window_minutes: int = 60
    cooldown_minutes: int = 30  # Minimum time between similar alerts
    
    # Feedback thresholds
    min_feedback_rating: float = 3.0
    negative_feedback_threshold: int = 3  # Number of negative feedbacks before alert
    
    # Notification settings
    email_enabled: bool = False
    slack_enabled: bool = False
    webhook_enabled: bool = False
    
    # Contact information
    email_recipients: List[str] = field(default_factory=list)
    slack_webhook_url: Optional[str] = None
    webhook_url: Optional[str] = None
    
    # SMTP settings
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True


class AlertingSystem:
    """
    Comprehensive alerting system that monitors quality metrics, performance,
    and user feedback to trigger appropriate alerts and notifications.
    """
    
    def __init__(self, config: AlertingConfig, log_dir: str = "logs"):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Feedback storage
        self.user_feedback: Dict[int, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # Violation tracking
        self.quality_violations: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_violations: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Notification handlers
        self.notification_handlers: List[Callable[[Alert], None]] = []
        
        # Setup logging
        self._setup_logging()
        
        # Setup default notification handlers
        self._setup_notification_handlers()
    
    def _setup_logging(self):
        """Setup logging for the alerting system"""
        log_file = self.log_dir / "alerts.log"
        
        self.logger = logging.getLogger("alerting_system")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler with rotation
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            log_file,
            maxBytes=20*1024*1024,  # 20MB
            backupCount=3
        )
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def _setup_notification_handlers(self):
        """Setup default notification handlers"""
        if self.config.email_enabled:
            self.notification_handlers.append(self._send_email_notification)
        
        if self.config.slack_enabled:
            self.notification_handlers.append(self._send_slack_notification)
        
        if self.config.webhook_enabled:
            self.notification_handlers.append(self._send_webhook_notification)
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a custom notification handler"""
        self.notification_handlers.append(handler)
    
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
        """
        Record user feedback and check for negative feedback patterns.
        
        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            query: Original query
            response: System response
            rating: User rating (1-5 scale)
            feedback_text: Optional feedback text
            session_id: Optional session identifier
            
        Returns:
            UserFeedback object
        """
        feedback = UserFeedback(
            tenant_id=tenant_id,
            user_id=user_id,
            query=query[:200],  # Truncate for storage
            response=response[:500],  # Truncate for storage
            rating=rating,
            feedback_text=feedback_text[:1000] if feedback_text else None,
            session_id=session_id
        )
        
        with self.lock:
            self.user_feedback[tenant_id].append(feedback)
        
        # Log feedback
        self.logger.info(f"USER_FEEDBACK: {json.dumps(feedback.to_dict())}")
        
        # Check for negative feedback patterns
        self._check_feedback_patterns(tenant_id)
        
        return feedback
    
    def check_quality_thresholds(self, metrics: QualityMetrics):
        """
        Check quality metrics against thresholds and trigger alerts if needed.
        
        Args:
            metrics: Quality metrics to check
        """
        violations = []
        
        # Check individual thresholds
        if metrics.retrieval_confidence < self.config.min_retrieval_confidence:
            violations.append({
                "type": "retrieval_confidence",
                "value": metrics.retrieval_confidence,
                "threshold": self.config.min_retrieval_confidence
            })
        
        if metrics.response_relevance < self.config.min_response_relevance:
            violations.append({
                "type": "response_relevance",
                "value": metrics.response_relevance,
                "threshold": self.config.min_response_relevance
            })
        
        if metrics.context_utilization < self.config.min_context_utilization:
            violations.append({
                "type": "context_utilization",
                "value": metrics.context_utilization,
                "threshold": self.config.min_context_utilization
            })
        
        if metrics.user_satisfaction_prediction < self.config.min_user_satisfaction:
            violations.append({
                "type": "user_satisfaction",
                "value": metrics.user_satisfaction_prediction,
                "threshold": self.config.min_user_satisfaction
            })
        
        if violations:
            with self.lock:
                self.quality_violations[metrics.tenant_id].append({
                    "timestamp": metrics.timestamp,
                    "violations": violations,
                    "query": metrics.query,
                    "user_id": metrics.user_id
                })
            
            # Check if we should trigger an alert
            self._check_quality_alert_conditions(metrics.tenant_id)
    
    def check_performance_thresholds(self, metrics: AggregatedMetrics):
        """
        Check performance metrics against thresholds and trigger alerts if needed.
        
        Args:
            metrics: Aggregated performance metrics to check
        """
        violations = []
        
        # Check performance thresholds
        if metrics.avg_response_time_ms > self.config.max_response_time_ms:
            violations.append({
                "type": "avg_response_time",
                "value": metrics.avg_response_time_ms,
                "threshold": self.config.max_response_time_ms
            })
        
        if metrics.p95_response_time_ms > self.config.max_p95_response_time_ms:
            violations.append({
                "type": "p95_response_time",
                "value": metrics.p95_response_time_ms,
                "threshold": self.config.max_p95_response_time_ms
            })
        
        if metrics.error_rate > self.config.max_error_rate:
            violations.append({
                "type": "error_rate",
                "value": metrics.error_rate,
                "threshold": self.config.max_error_rate
            })
        
        if violations:
            with self.lock:
                self.performance_violations[metrics.tenant_id].append({
                    "timestamp": datetime.now(),
                    "violations": violations,
                    "metrics": metrics.to_dict()
                })
            
            # Check if we should trigger an alert
            self._check_performance_alert_conditions(metrics.tenant_id)
    
    def _check_quality_alert_conditions(self, tenant_id: int):
        """Check if quality alert conditions are met"""
        with self.lock:
            violations = self.quality_violations[tenant_id]
            
            if len(violations) < self.config.violation_threshold:
                return
            
            # Check if violations occurred within the alert window
            now = datetime.now()
            window_start = now - timedelta(minutes=self.config.alert_window_minutes)
            
            recent_violations = [
                v for v in violations 
                if v["timestamp"] >= window_start
            ]
            
            if len(recent_violations) >= self.config.violation_threshold:
                self._create_quality_alert(tenant_id, recent_violations)
    
    def _check_performance_alert_conditions(self, tenant_id: int):
        """Check if performance alert conditions are met"""
        with self.lock:
            violations = self.performance_violations[tenant_id]
            
            if len(violations) < self.config.violation_threshold:
                return
            
            # Check if violations occurred within the alert window
            now = datetime.now()
            window_start = now - timedelta(minutes=self.config.alert_window_minutes)
            
            recent_violations = [
                v for v in violations 
                if v["timestamp"] >= window_start
            ]
            
            if len(recent_violations) >= self.config.violation_threshold:
                self._create_performance_alert(tenant_id, recent_violations)
    
    def _check_feedback_patterns(self, tenant_id: int):
        """Check for negative feedback patterns"""
        with self.lock:
            feedback_list = list(self.user_feedback[tenant_id])
        
        if len(feedback_list) < self.config.negative_feedback_threshold:
            return
        
        # Check recent feedback
        now = datetime.now()
        window_start = now - timedelta(minutes=self.config.alert_window_minutes)
        
        recent_feedback = [
            f for f in feedback_list 
            if f.timestamp >= window_start
        ]
        
        # Count negative feedback (rating <= min_feedback_rating)
        negative_count = sum(
            1 for f in recent_feedback 
            if f.rating <= self.config.min_feedback_rating
        )
        
        if negative_count >= self.config.negative_feedback_threshold:
            self._create_feedback_alert(tenant_id, recent_feedback)
    
    def _create_quality_alert(self, tenant_id: int, violations: List[Dict]):
        """Create a quality threshold alert"""
        alert_id = f"quality_{tenant_id}_{int(time.time())}"
        
        # Check cooldown
        cooldown_key = f"quality_{tenant_id}"
        if self._is_in_cooldown(cooldown_key):
            return
        
        # Determine severity based on violation types and frequency
        severity = self._determine_quality_severity(violations)
        
        # Create alert
        alert = Alert(
            id=alert_id,
            tenant_id=tenant_id,
            alert_type=AlertType.QUALITY_THRESHOLD,
            severity=severity,
            title=f"Quality Threshold Violations - Tenant {tenant_id}",
            description=self._format_quality_alert_description(violations),
            timestamp=datetime.now(),
            metadata={
                "violation_count": len(violations),
                "violations": violations[-5:]  # Last 5 violations
            }
        )
        
        self._trigger_alert(alert, cooldown_key)
    
    def _create_performance_alert(self, tenant_id: int, violations: List[Dict]):
        """Create a performance threshold alert"""
        alert_id = f"performance_{tenant_id}_{int(time.time())}"
        
        # Check cooldown
        cooldown_key = f"performance_{tenant_id}"
        if self._is_in_cooldown(cooldown_key):
            return
        
        # Determine severity
        severity = self._determine_performance_severity(violations)
        
        # Create alert
        alert = Alert(
            id=alert_id,
            tenant_id=tenant_id,
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=severity,
            title=f"Performance Degradation - Tenant {tenant_id}",
            description=self._format_performance_alert_description(violations),
            timestamp=datetime.now(),
            metadata={
                "violation_count": len(violations),
                "violations": violations[-3:]  # Last 3 violations
            }
        )
        
        self._trigger_alert(alert, cooldown_key)
    
    def _create_feedback_alert(self, tenant_id: int, feedback_list: List[UserFeedback]):
        """Create a negative feedback alert"""
        alert_id = f"feedback_{tenant_id}_{int(time.time())}"
        
        # Check cooldown
        cooldown_key = f"feedback_{tenant_id}"
        if self._is_in_cooldown(cooldown_key):
            return
        
        negative_feedback = [f for f in feedback_list if f.rating <= self.config.min_feedback_rating]
        avg_rating = sum(f.rating for f in feedback_list) / len(feedback_list)
        
        # Create alert
        alert = Alert(
            id=alert_id,
            tenant_id=tenant_id,
            alert_type=AlertType.USER_FEEDBACK_NEGATIVE,
            severity=AlertSeverity.MEDIUM,
            title=f"Negative User Feedback Pattern - Tenant {tenant_id}",
            description=f"Received {len(negative_feedback)} negative feedback entries "
                       f"(rating ≤ {self.config.min_feedback_rating}) in the last "
                       f"{self.config.alert_window_minutes} minutes. "
                       f"Average rating: {avg_rating:.2f}",
            timestamp=datetime.now(),
            metadata={
                "negative_count": len(negative_feedback),
                "total_feedback": len(feedback_list),
                "average_rating": avg_rating,
                "recent_feedback": [f.to_dict() for f in negative_feedback[-3:]]
            }
        )
        
        self._trigger_alert(alert, cooldown_key)
    
    def _determine_quality_severity(self, violations: List[Dict]) -> AlertSeverity:
        """Determine alert severity based on quality violations"""
        violation_count = len(violations)
        
        # Check for critical violations
        critical_types = {"user_satisfaction", "response_relevance"}
        critical_violations = sum(
            1 for v in violations 
            for violation in v["violations"]
            if violation["type"] in critical_types and violation["value"] < 0.3
        )
        
        if critical_violations > 0 or violation_count >= 10:
            return AlertSeverity.CRITICAL
        elif violation_count >= 7:
            return AlertSeverity.HIGH
        elif violation_count >= 5:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _determine_performance_severity(self, violations: List[Dict]) -> AlertSeverity:
        """Determine alert severity based on performance violations"""
        violation_count = len(violations)
        
        # Check for critical performance issues
        has_error_rate_violation = any(
            violation["type"] == "error_rate" and violation["value"] > 0.2
            for v in violations
            for violation in v["violations"]
        )
        
        has_severe_latency = any(
            violation["type"] in ["avg_response_time", "p95_response_time"] 
            and violation["value"] > 10000
            for v in violations
            for violation in v["violations"]
        )
        
        if has_error_rate_violation or has_severe_latency:
            return AlertSeverity.CRITICAL
        elif violation_count >= 5:
            return AlertSeverity.HIGH
        elif violation_count >= 3:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _format_quality_alert_description(self, violations: List[Dict]) -> str:
        """Format quality alert description"""
        violation_summary = defaultdict(int)
        
        for v in violations:
            for violation in v["violations"]:
                violation_summary[violation["type"]] += 1
        
        description = f"Quality threshold violations detected:\n"
        for violation_type, count in violation_summary.items():
            description += f"- {violation_type.replace('_', ' ').title()}: {count} violations\n"
        
        description += f"\nTotal violations in last {self.config.alert_window_minutes} minutes: {len(violations)}"
        
        return description
    
    def _format_performance_alert_description(self, violations: List[Dict]) -> str:
        """Format performance alert description"""
        violation_summary = defaultdict(int)
        
        for v in violations:
            for violation in v["violations"]:
                violation_summary[violation["type"]] += 1
        
        description = f"Performance threshold violations detected:\n"
        for violation_type, count in violation_summary.items():
            description += f"- {violation_type.replace('_', ' ').title()}: {count} violations\n"
        
        description += f"\nTotal violations in last {self.config.alert_window_minutes} minutes: {len(violations)}"
        
        return description
    
    def _is_in_cooldown(self, cooldown_key: str) -> bool:
        """Check if alert type is in cooldown period"""
        if cooldown_key not in self.alert_cooldowns:
            return False
        
        cooldown_end = self.alert_cooldowns[cooldown_key] + timedelta(
            minutes=self.config.cooldown_minutes
        )
        
        return datetime.now() < cooldown_end
    
    def _trigger_alert(self, alert: Alert, cooldown_key: str):
        """Trigger an alert and send notifications"""
        with self.lock:
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            self.alert_cooldowns[cooldown_key] = datetime.now()
        
        # Log alert
        self.logger.error(f"ALERT_TRIGGERED: {json.dumps(alert.to_dict())}")
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Failed to send notification: {e}")
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification for alert"""
        if not self.config.email_recipients or not self.config.smtp_server:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
Alert Details:
- ID: {alert.id}
- Tenant: {alert.tenant_id}
- Type: {alert.alert_type.value}
- Severity: {alert.severity.value}
- Timestamp: {alert.timestamp.isoformat()}

Description:
{alert.description}

Metadata:
{json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            if self.config.smtp_use_tls:
                server.starttls()
            if self.config.smtp_username and self.config.smtp_password:
                server.login(self.config.smtp_username, self.config.smtp_password)
            
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    def _send_slack_notification(self, alert: Alert):
        """Send Slack notification for alert"""
        if not self.config.slack_webhook_url:
            return
        
        try:
            color_map = {
                AlertSeverity.LOW: "#36a64f",      # Green
                AlertSeverity.MEDIUM: "#ff9500",   # Orange
                AlertSeverity.HIGH: "#ff0000",     # Red
                AlertSeverity.CRITICAL: "#8B0000"  # Dark Red
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#ff0000"),
                    "title": alert.title,
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "Alert ID",
                            "value": alert.id,
                            "short": True
                        },
                        {
                            "title": "Tenant ID",
                            "value": str(alert.tenant_id),
                            "short": True
                        },
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Type",
                            "value": alert.alert_type.value.replace('_', ' ').title(),
                            "short": True
                        }
                    ],
                    "timestamp": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            self.logger.info(f"Slack notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification for alert"""
        if not self.config.webhook_url:
            return
        
        try:
            payload = alert.to_dict()
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            self.logger.info(f"Webhook notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """
        Resolve an active alert.
        
        Args:
            alert_id: Alert identifier
            resolved_by: Who resolved the alert
            
        Returns:
            True if alert was resolved, False if not found
        """
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                alert.resolved_by = resolved_by
                
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
        
        return False
    
    def get_active_alerts(self, tenant_id: int = None) -> List[Alert]:
        """Get active alerts, optionally filtered by tenant"""
        with self.lock:
            alerts = list(self.active_alerts.values())
        
        if tenant_id is not None:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(self, tenant_id: int = None, limit: int = 50) -> List[Alert]:
        """Get alert history, optionally filtered by tenant"""
        with self.lock:
            alerts = list(self.alert_history)
        
        if tenant_id is not None:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]
        
        # Sort by timestamp (most recent first) and limit
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]
    
    def get_feedback_summary(self, tenant_id: int, hours: int = 24) -> Dict[str, Any]:
        """Get feedback summary for a tenant"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        with self.lock:
            feedback_list = [
                f for f in self.user_feedback[tenant_id]
                if f.timestamp >= start_time
            ]
        
        if not feedback_list:
            return {
                "tenant_id": tenant_id,
                "time_period": f"Last {hours} hours",
                "total_feedback": 0,
                "average_rating": 0.0,
                "rating_distribution": {},
                "negative_feedback_count": 0
            }
        
        total_feedback = len(feedback_list)
        average_rating = sum(f.rating for f in feedback_list) / total_feedback
        
        # Rating distribution
        rating_distribution = defaultdict(int)
        for f in feedback_list:
            rating_distribution[f.rating] += 1
        
        negative_feedback_count = sum(
            1 for f in feedback_list 
            if f.rating <= self.config.min_feedback_rating
        )
        
        return {
            "tenant_id": tenant_id,
            "time_period": f"Last {hours} hours",
            "total_feedback": total_feedback,
            "average_rating": average_rating,
            "rating_distribution": dict(rating_distribution),
            "negative_feedback_count": negative_feedback_count,
            "negative_feedback_rate": negative_feedback_count / total_feedback
        }
    
    def export_alerts(self, filepath: str, tenant_id: int = None, days: int = 7):
        """Export alert history to JSON file"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        with self.lock:
            alerts = [
                a for a in self.alert_history
                if a.timestamp >= start_time
            ]
        
        if tenant_id is not None:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "tenant_id": tenant_id,
            "time_period": f"Last {days} days",
            "total_alerts": len(alerts),
            "alerts": [a.to_dict() for a in alerts]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported {len(alerts)} alerts to {filepath}")


# Global alerting system instance
_alerting_system = None
_alerting_lock = threading.Lock()


def get_alerting_system(config: AlertingConfig = None, log_dir: str = "logs") -> AlertingSystem:
    """
    Get or create the global alerting system instance.
    
    Args:
        config: Alerting configuration
        log_dir: Directory for log files
        
    Returns:
        AlertingSystem instance
    """
    global _alerting_system
    
    with _alerting_lock:
        if _alerting_system is None:
            if config is None:
                config = AlertingConfig()
            _alerting_system = AlertingSystem(config=config, log_dir=log_dir)
        
        return _alerting_system


def reset_alerting_system():
    """Reset the global alerting system instance (mainly for testing)"""
    global _alerting_system
    
    with _alerting_lock:
        _alerting_system = None