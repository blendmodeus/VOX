"""
VØX Release - Reflect
---------------------

Reflection and continuous improvement analysis.

AXIØM Phase 12: Release/Reflect - "How do we ship this and learn from it?"
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .models import (
    ReflectionCategory,
    ReflectionInsight,
    ReflectionReport,
)

logger = logging.getLogger(__name__)


@dataclass
class ReflectionConfig:
    """
    Configuration for reflection engine.

    Attributes:
        metrics_path: Path to metrics data
        logs_path: Path to log files
        period_days: Analysis period in days
        include_recommendations: Generate recommendations
    """
    metrics_path: str = "metrics"
    logs_path: str = "logs"
    period_days: int = 30
    include_recommendations: bool = True


@dataclass
class UsageMetrics:
    """Usage metrics for reflection."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    unique_users: int = 0
    total_audio_seconds: float = 0.0
    peak_concurrent: int = 0
    endpoints_used: Dict[str, int] = field(default_factory=dict)
    voices_used: Dict[str, int] = field(default_factory=dict)
    features_used: Dict[str, int] = field(default_factory=dict)

    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate(),
            "unique_users": self.unique_users,
            "total_audio_seconds": self.total_audio_seconds,
            "peak_concurrent": self.peak_concurrent,
            "endpoints_used": self.endpoints_used,
            "voices_used": self.voices_used,
            "features_used": self.features_used,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for reflection."""
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_throughput_rps: float = 0.0
    peak_throughput_rps: float = 0.0
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latency": {
                "avg_ms": self.avg_latency_ms,
                "p50_ms": self.p50_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "p99_ms": self.p99_latency_ms,
                "max_ms": self.max_latency_ms,
            },
            "throughput": {
                "avg_rps": self.avg_throughput_rps,
                "peak_rps": self.peak_throughput_rps,
            },
            "resources": {
                "avg_memory_mb": self.avg_memory_mb,
                "peak_memory_mb": self.peak_memory_mb,
                "avg_cpu_percent": self.avg_cpu_percent,
                "peak_cpu_percent": self.peak_cpu_percent,
            },
        }


@dataclass
class ErrorMetrics:
    """Error metrics for reflection."""
    total_errors: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)
    error_endpoints: Dict[str, int] = field(default_factory=dict)
    error_trend: str = "stable"  # "increasing", "decreasing", "stable"
    mttr_hours: float = 0.0  # Mean time to resolution

    def error_rate(self, total_requests: int) -> float:
        """Calculate error rate."""
        if total_requests == 0:
            return 0.0
        return self.total_errors / total_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_errors": self.total_errors,
            "error_types": self.error_types,
            "error_endpoints": self.error_endpoints,
            "error_trend": self.error_trend,
            "mttr_hours": self.mttr_hours,
        }


class ReflectionEngine:
    """
    Engine for reflecting on system performance and usage.

    Features:
        - Usage pattern analysis
        - Performance trend analysis
        - Error pattern detection
        - Improvement recommendations
    """

    def __init__(
        self,
        config: Optional[ReflectionConfig] = None,
    ):
        """
        Initialize reflection engine.

        Args:
            config: Reflection configuration
        """
        self.config = config or ReflectionConfig()
        self._usage: Optional[UsageMetrics] = None
        self._performance: Optional[PerformanceMetrics] = None
        self._errors: Optional[ErrorMetrics] = None

    def generate_report(
        self,
        version: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> ReflectionReport:
        """
        Generate comprehensive reflection report.

        Args:
            version: Version to analyze
            period_start: Start of analysis period
            period_end: End of analysis period

        Returns:
            Reflection report
        """
        if not period_end:
            period_end = datetime.now()
        if not period_start:
            period_start = period_end - timedelta(days=self.config.period_days)

        report = ReflectionReport(
            version=version,
            period_start=period_start,
            period_end=period_end,
        )

        # Load metrics
        self._load_metrics(period_start, period_end)

        # Generate usage summary
        if self._usage:
            report.usage_summary = self._usage.to_dict()

        # Generate performance summary
        if self._performance:
            report.performance_summary = self._performance.to_dict()

        # Generate error summary
        if self._errors:
            report.error_summary = self._errors.to_dict()

        # Generate insights
        report.insights = self._generate_insights()

        # Generate recommendations
        if self.config.include_recommendations:
            report.recommendations = self._generate_recommendations()

        return report

    def analyze_usage_patterns(self) -> List[ReflectionInsight]:
        """
        Analyze usage patterns.

        Returns:
            List of insights
        """
        insights = []

        if not self._usage:
            return insights

        # Check success rate
        success_rate = self._usage.success_rate()
        if success_rate < 0.95:
            insights.append(ReflectionInsight(
                category=ReflectionCategory.RELIABILITY,
                title="Low Success Rate",
                description=f"Success rate is {success_rate:.1%}, below 95% threshold.",
                impact="high",
                recommendation="Investigate error causes and improve error handling.",
                metrics={"success_rate": success_rate},
            ))
        elif success_rate > 0.99:
            insights.append(ReflectionInsight(
                category=ReflectionCategory.RELIABILITY,
                title="Excellent Success Rate",
                description=f"Success rate is {success_rate:.1%}, above 99% threshold.",
                impact="low",
                metrics={"success_rate": success_rate},
            ))

        # Check feature adoption
        if self._usage.features_used:
            unused_features = [
                f for f, count in self._usage.features_used.items()
                if count == 0
            ]
            if unused_features:
                insights.append(ReflectionInsight(
                    category=ReflectionCategory.USABILITY,
                    title="Unused Features Detected",
                    description=f"{len(unused_features)} features have no usage.",
                    impact="medium",
                    recommendation="Review feature discoverability or consider deprecation.",
                    metrics={"unused_features": unused_features[:5]},
                ))

        # Check voice distribution
        if self._usage.voices_used:
            total_voice_uses = sum(self._usage.voices_used.values())
            if total_voice_uses > 0:
                top_voice = max(self._usage.voices_used.items(), key=lambda x: x[1])
                if top_voice[1] / total_voice_uses > 0.8:
                    insights.append(ReflectionInsight(
                        category=ReflectionCategory.USABILITY,
                        title="Voice Usage Concentration",
                        description=f"Voice '{top_voice[0]}' accounts for {top_voice[1]/total_voice_uses:.0%} of usage.",
                        impact="low",
                        recommendation="Consider featuring other voices or understanding user preferences.",
                    ))

        return insights

    def analyze_performance_trends(self) -> List[ReflectionInsight]:
        """
        Analyze performance trends.

        Returns:
            List of insights
        """
        insights = []

        if not self._performance:
            return insights

        # Check latency
        if self._performance.p95_latency_ms > 1000:
            insights.append(ReflectionInsight(
                category=ReflectionCategory.PERFORMANCE,
                title="High P95 Latency",
                description=f"P95 latency is {self._performance.p95_latency_ms:.0f}ms, above 1000ms threshold.",
                impact="high",
                recommendation="Profile slow requests and optimize critical paths.",
                metrics={
                    "p95_latency_ms": self._performance.p95_latency_ms,
                    "avg_latency_ms": self._performance.avg_latency_ms,
                },
            ))
        elif self._performance.p95_latency_ms < 200:
            insights.append(ReflectionInsight(
                category=ReflectionCategory.PERFORMANCE,
                title="Excellent Latency",
                description=f"P95 latency is {self._performance.p95_latency_ms:.0f}ms, well under 200ms.",
                impact="low",
                metrics={"p95_latency_ms": self._performance.p95_latency_ms},
            ))

        # Check memory usage
        if self._performance.peak_memory_mb > 1024:
            insights.append(ReflectionInsight(
                category=ReflectionCategory.PERFORMANCE,
                title="High Memory Usage",
                description=f"Peak memory usage is {self._performance.peak_memory_mb:.0f}MB.",
                impact="medium",
                recommendation="Review memory allocation patterns and consider optimization.",
                metrics={
                    "peak_memory_mb": self._performance.peak_memory_mb,
                    "avg_memory_mb": self._performance.avg_memory_mb,
                },
            ))

        # Check throughput
        if self._performance.peak_throughput_rps > 0:
            capacity_ratio = self._performance.avg_throughput_rps / self._performance.peak_throughput_rps
            if capacity_ratio > 0.8:
                insights.append(ReflectionInsight(
                    category=ReflectionCategory.PERFORMANCE,
                    title="Near Capacity",
                    description=f"Average throughput is {capacity_ratio:.0%} of peak capacity.",
                    impact="high",
                    recommendation="Consider scaling or optimizing to handle traffic spikes.",
                ))

        return insights

    def analyze_error_patterns(self) -> List[ReflectionInsight]:
        """
        Analyze error patterns.

        Returns:
            List of insights
        """
        insights = []

        if not self._errors:
            return insights

        # Check error trend
        if self._errors.error_trend == "increasing":
            insights.append(ReflectionInsight(
                category=ReflectionCategory.RELIABILITY,
                title="Increasing Error Rate",
                description="Error rate is trending upward over the analysis period.",
                impact="high",
                recommendation="Investigate root cause and implement fixes urgently.",
            ))

        # Check error concentration
        if self._errors.error_types:
            top_error = max(self._errors.error_types.items(), key=lambda x: x[1])
            if top_error[1] > self._errors.total_errors * 0.5:
                insights.append(ReflectionInsight(
                    category=ReflectionCategory.RELIABILITY,
                    title="Concentrated Error Type",
                    description=f"'{top_error[0]}' accounts for {top_error[1]/self._errors.total_errors:.0%} of errors.",
                    impact="high",
                    recommendation=f"Focus on fixing {top_error[0]} errors for maximum impact.",
                    metrics={"error_type": top_error[0], "count": top_error[1]},
                ))

        # Check MTTR
        if self._errors.mttr_hours > 24:
            insights.append(ReflectionInsight(
                category=ReflectionCategory.RELIABILITY,
                title="High Mean Time to Resolution",
                description=f"Average error resolution time is {self._errors.mttr_hours:.1f} hours.",
                impact="medium",
                recommendation="Improve incident response procedures and monitoring.",
            ))

        return insights

    def analyze_security(self) -> List[ReflectionInsight]:
        """
        Analyze security posture.

        Returns:
            List of insights
        """
        insights = []

        # Check for security-related features
        insights.append(ReflectionInsight(
            category=ReflectionCategory.SECURITY,
            title="Security Features Review",
            description="Review of security features and best practices.",
            impact="medium",
            recommendation="Ensure rate limiting, authentication, and input validation are enabled.",
        ))

        return insights

    def _load_metrics(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> None:
        """Load metrics from storage."""
        # In a real implementation, this would load from metrics storage
        # For now, create sample metrics

        self._usage = UsageMetrics(
            total_requests=10000,
            successful_requests=9850,
            failed_requests=150,
            unique_users=500,
            total_audio_seconds=50000.0,
            peak_concurrent=50,
            endpoints_used={
                "/synthesize": 8000,
                "/stream": 1500,
                "/voices": 500,
            },
            voices_used={
                "default": 6000,
                "narrator": 2000,
                "assistant": 2000,
            },
            features_used={
                "streaming": 1500,
                "multi_voice": 500,
                "biometrics": 100,
            },
        )

        self._performance = PerformanceMetrics(
            avg_latency_ms=150.0,
            p50_latency_ms=120.0,
            p95_latency_ms=350.0,
            p99_latency_ms=800.0,
            max_latency_ms=2500.0,
            avg_throughput_rps=5.0,
            peak_throughput_rps=25.0,
            avg_memory_mb=256.0,
            peak_memory_mb=512.0,
            avg_cpu_percent=15.0,
            peak_cpu_percent=75.0,
        )

        self._errors = ErrorMetrics(
            total_errors=150,
            error_types={
                "ValidationError": 80,
                "TimeoutError": 40,
                "RateLimitError": 20,
                "InternalError": 10,
            },
            error_endpoints={
                "/synthesize": 100,
                "/stream": 40,
                "/biometrics/verify": 10,
            },
            error_trend="stable",
            mttr_hours=4.5,
        )

    def _generate_insights(self) -> List[ReflectionInsight]:
        """Generate all insights."""
        insights = []

        insights.extend(self.analyze_usage_patterns())
        insights.extend(self.analyze_performance_trends())
        insights.extend(self.analyze_error_patterns())
        insights.extend(self.analyze_security())

        # Sort by impact
        impact_order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda i: impact_order.get(i.impact, 2))

        return insights

    def _generate_recommendations(self) -> List[str]:
        """Generate top recommendations."""
        recommendations = []

        # Based on insights
        insights = self._generate_insights()
        high_impact = [i for i in insights if i.impact == "high"]

        for insight in high_impact[:3]:
            if insight.recommendation:
                recommendations.append(insight.recommendation)

        # Add general recommendations
        recommendations.extend([
            "Continue monitoring key metrics and set up alerting for anomalies.",
            "Document any lessons learned for future reference.",
            "Plan the next iteration based on user feedback and metrics.",
        ])

        return recommendations[:5]


def generate_reflection_report(
    version: str,
    period_days: int = 30,
) -> ReflectionReport:
    """
    Generate VØX reflection report.

    Args:
        version: Version to analyze
        period_days: Analysis period

    Returns:
        Reflection report
    """
    config = ReflectionConfig(period_days=period_days)
    engine = ReflectionEngine(config)
    return engine.generate_report(version)


def print_reflection_report(version: str) -> None:
    """Print reflection report to stdout."""
    report = generate_reflection_report(version)
    print(report.to_markdown())
