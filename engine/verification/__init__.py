"""
VØX Verification Layer
----------------------

Comprehensive verification suite for VØX.

Features:
    - End-to-end testing framework
    - Performance benchmarking
    - Audio quality validation
    - System health monitoring
    - Liveness and readiness probes

Quick Start:
    >>> from axiom_vox.verification import (
    ...     E2ETestRunner, BenchmarkSuite,
    ...     QualityValidator, HealthChecker,
    ... )
    >>>
    >>> # Run E2E tests
    >>> runner = E2ETestRunner()
    >>> results = await runner.run_all()
    >>>
    >>> # Run benchmarks
    >>> suite = BenchmarkSuite()
    >>> benchmarks = await suite.run_all()
    >>>
    >>> # Check quality
    >>> validator = QualityValidator()
    >>> quality = validator.validate(audio_bytes)
    >>>
    >>> # Health check
    >>> checker = HealthChecker()
    >>> health = await checker.check_health()

AXIØM Phase 10: Verify - "How do we know this works?"
"""

from .models import (
    # Enums
    TestStatus,
    TestSeverity,
    BenchmarkType,
    HealthStatus,
    # Test models
    TestCase,
    TestResult,
    TestSuiteResult,
    # Benchmark models
    BenchmarkMetric,
    BenchmarkResult,
    # Quality models
    QualityMetric,
    QualityResult,
    # Health models
    HealthCheckResult,
    SystemHealthReport,
    # Report
    VerificationReport,
)

from .e2e import (
    E2ETestConfig,
    E2ETestRunner,
    create_vox_e2e_tests,
    run_quick_verification,
    run_full_verification,
)

from .benchmark import (
    BenchmarkConfig,
    BenchmarkDefinition,
    BenchmarkSuite,
    create_vox_benchmarks,
    run_benchmarks,
    run_latency_benchmarks,
)

from .quality import (
    QualityThresholds,
    QualityValidator,
    IntelligibilityEstimator,
    create_quality_validator,
)

from .health import (
    HealthCheckConfig,
    HealthCheckDefinition,
    HealthChecker,
    LivenessProbe,
    ReadinessProbe,
    create_vox_health_checker,
    run_health_check,
    check_system_ready,
)


__all__ = [
    # Enums
    "TestStatus",
    "TestSeverity",
    "BenchmarkType",
    "HealthStatus",
    # Test models
    "TestCase",
    "TestResult",
    "TestSuiteResult",
    # Benchmark models
    "BenchmarkMetric",
    "BenchmarkResult",
    # Quality models
    "QualityMetric",
    "QualityResult",
    # Health models
    "HealthCheckResult",
    "SystemHealthReport",
    # Report
    "VerificationReport",
    # E2E testing
    "E2ETestConfig",
    "E2ETestRunner",
    "create_vox_e2e_tests",
    "run_quick_verification",
    "run_full_verification",
    # Benchmarking
    "BenchmarkConfig",
    "BenchmarkDefinition",
    "BenchmarkSuite",
    "create_vox_benchmarks",
    "run_benchmarks",
    "run_latency_benchmarks",
    # Quality
    "QualityThresholds",
    "QualityValidator",
    "IntelligibilityEstimator",
    "create_quality_validator",
    # Health
    "HealthCheckConfig",
    "HealthCheckDefinition",
    "HealthChecker",
    "LivenessProbe",
    "ReadinessProbe",
    "create_vox_health_checker",
    "run_health_check",
    "check_system_ready",
]
