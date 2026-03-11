"""
VØX Verification - Models
-------------------------

Data models for testing and verification.

AXIØM Phase 10: Verify - "How do we know this works?"
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Union


class TestStatus(str, Enum):
    """Status of a test case."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestSeverity(str, Enum):
    """Severity level of a test failure."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class BenchmarkType(str, Enum):
    """Type of benchmark."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    QUALITY = "quality"


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class TestCase:
    """
    Definition of a test case.

    Attributes:
        name: Test name
        description: What the test verifies
        category: Test category (e.g., "synthesis", "streaming")
        severity: Impact severity if test fails
        timeout: Max execution time in seconds
        tags: Additional tags for filtering
        setup: Optional setup function
        teardown: Optional teardown function
    """
    name: str
    description: str
    category: str = "general"
    severity: TestSeverity = TestSeverity.MEDIUM
    timeout: float = 30.0
    tags: List[str] = field(default_factory=list)
    setup: Optional[Callable] = None
    teardown: Optional[Callable] = None
    enabled: bool = True

    @property
    def id(self) -> str:
        """Generate unique test ID."""
        return hashlib.md5(
            f"{self.category}:{self.name}".encode()
        ).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "severity": self.severity.value,
            "timeout": self.timeout,
            "tags": self.tags,
            "enabled": self.enabled,
        }


@dataclass
class TestResult:
    """
    Result of a single test execution.

    Attributes:
        test_case: The test that was run
        status: Pass/fail status
        duration_ms: Execution time
        message: Success/failure message
        error: Error details if failed
        artifacts: Output artifacts (logs, files)
        metrics: Measured metrics
    """
    test_case: TestCase
    status: TestStatus
    duration_ms: float
    message: str = ""
    error: Optional[str] = None
    error_trace: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == TestStatus.PASSED

    @property
    def failed(self) -> bool:
        """Check if test failed."""
        return self.status in (TestStatus.FAILED, TestStatus.ERROR)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_case.id,
            "test_name": self.test_case.name,
            "category": self.test_case.category,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "message": self.message,
            "error": self.error,
            "metrics": self.metrics,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class TestSuiteResult:
    """
    Result of running a test suite.

    Attributes:
        name: Suite name
        results: Individual test results
        total_duration_ms: Total execution time
        passed: Number of passed tests
        failed: Number of failed tests
        skipped: Number of skipped tests
    """
    name: str
    results: List[TestResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        """Count passed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        """Count failed tests."""
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def errors(self) -> int:
        """Count error tests."""
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    @property
    def skipped(self) -> int:
        """Count skipped tests."""
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def total(self) -> int:
        """Total number of tests."""
        return len(self.results)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.errors == 0

    def add_result(self, result: TestResult) -> None:
        """Add a test result."""
        self.results.append(result)
        self.total_duration_ms += result.duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "success_rate": self.success_rate,
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Test Suite: {self.name}",
            f"  Total: {self.total}",
            f"  Passed: {self.passed}",
            f"  Failed: {self.failed}",
            f"  Errors: {self.errors}",
            f"  Skipped: {self.skipped}",
            f"  Success Rate: {self.success_rate:.1%}",
            f"  Duration: {self.total_duration_ms:.1f}ms",
        ]
        return "\n".join(lines)


@dataclass
class BenchmarkMetric:
    """
    A single benchmark metric measurement.

    Attributes:
        name: Metric name
        value: Measured value
        unit: Unit of measurement
        threshold: Optional pass/fail threshold
    """
    name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    threshold_type: str = "max"  # "max", "min"

    @property
    def passed(self) -> bool:
        """Check if metric passes threshold."""
        if self.threshold is None:
            return True
        if self.threshold_type == "max":
            return self.value <= self.threshold
        else:
            return self.value >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "threshold": self.threshold,
            "threshold_type": self.threshold_type,
            "passed": self.passed,
        }


@dataclass
class BenchmarkResult:
    """
    Result of a benchmark run.

    Attributes:
        name: Benchmark name
        benchmark_type: Type of benchmark
        metrics: Measured metrics
        iterations: Number of iterations run
        warmup_iterations: Warmup iterations
    """
    name: str
    benchmark_type: BenchmarkType
    metrics: List[BenchmarkMetric] = field(default_factory=list)
    iterations: int = 1
    warmup_iterations: int = 0
    duration_ms: float = 0.0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if all metrics pass thresholds."""
        return all(m.passed for m in self.metrics)

    def add_metric(
        self,
        name: str,
        value: float,
        unit: str,
        threshold: Optional[float] = None,
        threshold_type: str = "max",
    ) -> BenchmarkMetric:
        """Add a metric measurement."""
        metric = BenchmarkMetric(
            name=name,
            value=value,
            unit=unit,
            threshold=threshold,
            threshold_type=threshold_type,
        )
        self.metrics.append(metric)
        return metric

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.benchmark_type.value,
            "passed": self.passed,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "duration_ms": self.duration_ms,
            "metrics": [m.to_dict() for m in self.metrics],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


@dataclass
class QualityMetric:
    """
    Audio quality metric.

    Attributes:
        name: Metric name (e.g., "snr", "intelligibility")
        value: Measured value
        min_threshold: Minimum acceptable value
        max_threshold: Maximum acceptable value
        weight: Weight for composite scoring
    """
    name: str
    value: float
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    weight: float = 1.0
    description: str = ""

    @property
    def passed(self) -> bool:
        """Check if metric is within thresholds."""
        if self.min_threshold is not None and self.value < self.min_threshold:
            return False
        if self.max_threshold is not None and self.value > self.max_threshold:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "weight": self.weight,
            "passed": self.passed,
            "description": self.description,
        }


@dataclass
class QualityResult:
    """
    Result of audio quality validation.

    Attributes:
        metrics: Individual quality metrics
        composite_score: Weighted overall score
        audio_duration_ms: Duration of analyzed audio
    """
    metrics: List[QualityMetric] = field(default_factory=list)
    composite_score: float = 0.0
    audio_duration_ms: float = 0.0
    sample_rate: int = 24000
    analysis_duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if all metrics pass."""
        return all(m.passed for m in self.metrics)

    def calculate_composite(self) -> float:
        """Calculate weighted composite score."""
        if not self.metrics:
            return 0.0

        total_weight = sum(m.weight for m in self.metrics)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(m.value * m.weight for m in self.metrics)
        self.composite_score = weighted_sum / total_weight
        return self.composite_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "composite_score": self.composite_score,
            "audio_duration_ms": self.audio_duration_ms,
            "sample_rate": self.sample_rate,
            "analysis_duration_ms": self.analysis_duration_ms,
            "metrics": [m.to_dict() for m in self.metrics],
            "metadata": self.metadata,
        }


@dataclass
class HealthCheckResult:
    """
    Result of a health check.

    Attributes:
        name: Check name
        status: Health status
        message: Status message
        latency_ms: Check latency
        details: Additional details
    """
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def healthy(self) -> bool:
        """Check if status is healthy."""
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "healthy": self.healthy,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class SystemHealthReport:
    """
    Overall system health report.

    Attributes:
        checks: Individual health checks
        overall_status: Aggregate status
        ready: Whether system is ready to serve
        live: Whether system is alive
    """
    checks: List[HealthCheckResult] = field(default_factory=list)
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    ready: bool = False
    live: bool = True
    version: str = ""
    uptime_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_overall(self) -> HealthStatus:
        """Calculate overall status from checks."""
        if not self.checks:
            self.overall_status = HealthStatus.UNKNOWN
            return self.overall_status

        statuses = [c.status for c in self.checks]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            self.overall_status = HealthStatus.HEALTHY
            self.ready = True
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            self.overall_status = HealthStatus.UNHEALTHY
            self.ready = False
        else:
            self.overall_status = HealthStatus.DEGRADED
            self.ready = True  # Degraded but still operational

        return self.overall_status

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_status": self.overall_status.value,
            "ready": self.ready,
            "live": self.live,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"System Health: {self.overall_status.value.upper()}",
            f"  Ready: {self.ready}",
            f"  Live: {self.live}",
            f"  Version: {self.version}",
            f"  Uptime: {self.uptime_seconds:.0f}s",
            "",
            "Checks:",
        ]
        for check in self.checks:
            status_icon = "✓" if check.healthy else "✗"
            lines.append(f"  {status_icon} {check.name}: {check.status.value}")

        return "\n".join(lines)


@dataclass
class VerificationReport:
    """
    Complete verification report combining all checks.

    Attributes:
        test_results: Test suite results
        benchmark_results: Benchmark results
        quality_results: Quality validation results
        health_report: System health report
    """
    test_results: Optional[TestSuiteResult] = None
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    quality_results: Optional[QualityResult] = None
    health_report: Optional[SystemHealthReport] = None
    generated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        """Check if all verification checks passed."""
        passed = True

        if self.test_results and not self.test_results.all_passed:
            passed = False

        if self.benchmark_results and not all(b.passed for b in self.benchmark_results):
            passed = False

        if self.quality_results and not self.quality_results.passed:
            passed = False

        if self.health_report and not self.health_report.ready:
            passed = False

        return passed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "all_passed": self.all_passed,
            "test_results": self.test_results.to_dict() if self.test_results else None,
            "benchmark_results": [b.to_dict() for b in self.benchmark_results],
            "quality_results": self.quality_results.to_dict() if self.quality_results else None,
            "health_report": self.health_report.to_dict() if self.health_report else None,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate complete verification summary."""
        lines = [
            "=" * 50,
            "VØX VERIFICATION REPORT",
            "=" * 50,
            "",
        ]

        if self.test_results:
            lines.append(self.test_results.summary())
            lines.append("")

        if self.benchmark_results:
            lines.append("Benchmarks:")
            for bench in self.benchmark_results:
                status = "PASS" if bench.passed else "FAIL"
                lines.append(f"  [{status}] {bench.name}")
            lines.append("")

        if self.quality_results:
            status = "PASS" if self.quality_results.passed else "FAIL"
            lines.append(f"Quality Check: [{status}]")
            lines.append(f"  Composite Score: {self.quality_results.composite_score:.2f}")
            lines.append("")

        if self.health_report:
            lines.append(self.health_report.summary())
            lines.append("")

        lines.append("=" * 50)
        overall = "ALL CHECKS PASSED" if self.all_passed else "SOME CHECKS FAILED"
        lines.append(f"RESULT: {overall}")
        lines.append("=" * 50)

        return "\n".join(lines)
