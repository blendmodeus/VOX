"""
VØX Verification Layer Tests
----------------------------

Comprehensive tests for the VØX Verification Layer.

AXIØM Phase 10: Verify - "How do we know this works?"
"""

import asyncio
import pytest
import time
import numpy as np


# ============================================================================
# Model Tests
# ============================================================================


class TestVerificationModels:
    """Tests for verification models."""

    def test_test_case(self):
        """Test TestCase model."""
        from axiom_vox.verification import TestCase, TestSeverity

        test = TestCase(
            name="test_example",
            description="An example test",
            category="example",
            severity=TestSeverity.HIGH,
            tags=["smoke", "core"],
        )

        assert test.name == "test_example"
        assert test.severity == TestSeverity.HIGH
        assert "smoke" in test.tags
        assert len(test.id) == 12  # MD5 hash prefix

    def test_test_result(self):
        """Test TestResult model."""
        from axiom_vox.verification import TestCase, TestResult, TestStatus

        test = TestCase(name="test_pass", description="A passing test")
        result = TestResult(
            test_case=test,
            status=TestStatus.PASSED,
            duration_ms=100.5,
            message="Test passed",
        )

        assert result.passed
        assert not result.failed
        assert result.duration_ms == 100.5

    def test_test_suite_result(self):
        """Test TestSuiteResult model."""
        from axiom_vox.verification import (
            TestCase,
            TestResult,
            TestSuiteResult,
            TestStatus,
        )

        suite = TestSuiteResult(name="Test Suite")

        # Add passing test
        test1 = TestCase(name="test1", description="Test 1")
        result1 = TestResult(test_case=test1, status=TestStatus.PASSED, duration_ms=50)
        suite.add_result(result1)

        # Add failing test
        test2 = TestCase(name="test2", description="Test 2")
        result2 = TestResult(test_case=test2, status=TestStatus.FAILED, duration_ms=30)
        suite.add_result(result2)

        assert suite.total == 2
        assert suite.passed == 1
        assert suite.failed == 1
        assert suite.success_rate == 0.5
        assert not suite.all_passed

    def test_benchmark_result(self):
        """Test BenchmarkResult model."""
        from axiom_vox.verification import BenchmarkResult, BenchmarkType

        result = BenchmarkResult(
            name="latency_benchmark",
            benchmark_type=BenchmarkType.LATENCY,
            iterations=10,
        )

        # Add metrics
        result.add_metric("latency_mean_ms", 100.0, "ms", threshold=200.0)
        result.add_metric("latency_p99_ms", 150.0, "ms", threshold=300.0)

        assert len(result.metrics) == 2
        assert result.passed  # All within thresholds

    def test_quality_result(self):
        """Test QualityResult model."""
        from axiom_vox.verification import QualityResult, QualityMetric

        result = QualityResult(
            audio_duration_ms=1000.0,
            sample_rate=24000,
        )

        result.metrics.append(QualityMetric(
            name="snr_db",
            value=25.0,
            min_threshold=20.0,
            weight=1.5,
        ))

        result.metrics.append(QualityMetric(
            name="clipping_ratio",
            value=0.001,
            max_threshold=0.01,
            weight=1.0,
        ))

        result.calculate_composite()

        assert result.passed
        assert result.composite_score > 0

    def test_health_check_result(self):
        """Test HealthCheckResult model."""
        from axiom_vox.verification import (
            HealthCheckResult,
            SystemHealthReport,
            HealthStatus,
        )

        check = HealthCheckResult(
            name="engine",
            status=HealthStatus.HEALTHY,
            message="Engine operational",
            latency_ms=50.0,
        )

        assert check.healthy
        assert check.latency_ms == 50.0

        # Test system report
        report = SystemHealthReport()
        report.checks.append(check)
        report.checks.append(HealthCheckResult(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connected",
        ))

        report.calculate_overall()
        assert report.overall_status == HealthStatus.HEALTHY
        assert report.ready


# ============================================================================
# E2E Testing Tests
# ============================================================================


class TestE2ETestRunner:
    """Tests for E2E test runner."""

    @pytest.mark.asyncio
    async def test_register_and_run_test(self):
        """Test registering and running a test."""
        from axiom_vox.verification import E2ETestRunner, TestStatus

        runner = E2ETestRunner()

        @runner.test(category="unit")
        async def test_simple():
            """A simple test."""
            assert 1 + 1 == 2

        result = await runner.run_all()

        assert result.total == 1
        assert result.passed == 1
        assert result.all_passed

    @pytest.mark.asyncio
    async def test_failing_test(self):
        """Test handling of failing tests."""
        from axiom_vox.verification import E2ETestRunner

        runner = E2ETestRunner()

        @runner.test(category="unit")
        async def test_failure():
            """A failing test."""
            assert 1 == 2, "Expected failure"

        result = await runner.run_all()

        assert result.total == 1
        assert result.failed == 1

    @pytest.mark.asyncio
    async def test_test_timeout(self):
        """Test timeout handling."""
        from axiom_vox.verification import E2ETestRunner, E2ETestConfig

        runner = E2ETestRunner(E2ETestConfig(timeout=0.1))

        @runner.test(timeout=0.1)
        async def test_slow():
            """A slow test."""
            await asyncio.sleep(1.0)

        result = await runner.run_all()

        assert result.failed == 1
        assert "timeout" in result.results[0].message.lower()

    @pytest.mark.asyncio
    async def test_category_filtering(self):
        """Test category-based filtering."""
        from axiom_vox.verification import E2ETestRunner

        runner = E2ETestRunner()

        @runner.test(category="fast")
        async def test_fast():
            pass

        @runner.test(category="slow")
        async def test_slow():
            pass

        result = await runner.run_by_category("fast")

        assert result.total == 1

    @pytest.mark.asyncio
    async def test_tag_filtering(self):
        """Test tag-based filtering."""
        from axiom_vox.verification import E2ETestRunner

        runner = E2ETestRunner()

        @runner.test(tags=["smoke"])
        async def test_smoke():
            pass

        @runner.test(tags=["integration"])
        async def test_integration():
            pass

        result = await runner.run_by_tags(["smoke"])

        assert result.total == 1

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Test progress callback."""
        from axiom_vox.verification import E2ETestRunner

        runner = E2ETestRunner()

        @runner.test()
        async def test1():
            pass

        @runner.test()
        async def test2():
            pass

        progress_updates = []

        def on_progress(completed, total, result):
            progress_updates.append((completed, total))

        await runner.run_all(on_progress=on_progress)

        assert len(progress_updates) == 2
        assert progress_updates[-1] == (2, 2)


# ============================================================================
# Benchmark Tests
# ============================================================================


class TestBenchmarkSuite:
    """Tests for benchmark suite."""

    @pytest.mark.asyncio
    async def test_register_and_run_benchmark(self):
        """Test registering and running a benchmark."""
        from axiom_vox.verification import BenchmarkSuite, BenchmarkType

        suite = BenchmarkSuite()

        @suite.benchmark(benchmark_type=BenchmarkType.LATENCY)
        async def simple_benchmark():
            """A simple benchmark."""
            await asyncio.sleep(0.01)

        results = await suite.run_all()

        assert len(results) == 1
        assert results[0].name == "simple_benchmark"
        assert len(results[0].metrics) > 0

    @pytest.mark.asyncio
    async def test_latency_metrics(self):
        """Test latency metric collection."""
        from axiom_vox.verification import (
            BenchmarkSuite,
            BenchmarkConfig,
            BenchmarkType,
        )

        suite = BenchmarkSuite(BenchmarkConfig(
            iterations=5,
            warmup_iterations=1,
            memory_tracking=False,
        ))

        @suite.benchmark(benchmark_type=BenchmarkType.LATENCY)
        async def latency_benchmark():
            await asyncio.sleep(0.01)

        results = await suite.run_all()
        result = results[0]

        # Check latency metrics exist
        metric_names = {m.name for m in result.metrics}
        assert "latency_mean_ms" in metric_names
        assert "latency_p95_ms" in metric_names

    @pytest.mark.asyncio
    async def test_threshold_validation(self):
        """Test threshold validation."""
        from axiom_vox.verification import BenchmarkSuite, BenchmarkType

        suite = BenchmarkSuite()

        @suite.benchmark(
            benchmark_type=BenchmarkType.LATENCY,
            thresholds={"latency_mean_ms": 1.0},  # Very low threshold
        )
        async def fast_benchmark():
            await asyncio.sleep(0.01)  # 10ms > 1ms threshold

        results = await suite.run_all()

        # Should fail threshold
        assert not results[0].passed

    @pytest.mark.asyncio
    async def test_run_by_type(self):
        """Test running benchmarks by type."""
        from axiom_vox.verification import BenchmarkSuite, BenchmarkType

        suite = BenchmarkSuite()

        @suite.benchmark(benchmark_type=BenchmarkType.LATENCY)
        async def latency_bench():
            pass

        @suite.benchmark(benchmark_type=BenchmarkType.THROUGHPUT)
        async def throughput_bench():
            pass

        latency_results = await suite.run_by_type(BenchmarkType.LATENCY)

        assert len(latency_results) == 1
        assert latency_results[0].name == "latency_bench"


# ============================================================================
# Quality Validation Tests
# ============================================================================


class TestQualityValidator:
    """Tests for quality validator."""

    def _generate_test_audio(
        self,
        duration_ms: float = 1000,
        sample_rate: int = 24000,
        frequency: float = 440.0,
        amplitude: float = 0.5,
    ) -> bytes:
        """Generate test audio."""
        samples = int(sample_rate * duration_ms / 1000)
        t = np.arange(samples) / sample_rate
        audio = amplitude * np.sin(2 * np.pi * frequency * t)

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def test_validate_good_audio(self):
        """Test validation of good quality audio."""
        from axiom_vox.verification import QualityValidator

        validator = QualityValidator()
        audio = self._generate_test_audio(amplitude=0.5)

        result = validator.validate(audio, sample_rate=24000)

        assert result.audio_duration_ms > 0
        assert len(result.metrics) > 0

    def test_validate_silent_audio(self):
        """Test detection of excessive silence."""
        from axiom_vox.verification import QualityValidator

        validator = QualityValidator()

        # Generate mostly silent audio
        audio = self._generate_test_audio(amplitude=0.001)

        result = validator.validate(audio)

        # Check silence metric
        silence_metric = next(
            (m for m in result.metrics if m.name == "silence_ratio"),
            None,
        )
        assert silence_metric is not None
        assert silence_metric.value > 0.5  # Mostly silent

    def test_validate_clipped_audio(self):
        """Test detection of audio clipping."""
        from axiom_vox.verification import QualityValidator

        validator = QualityValidator()

        # Generate clipped audio (amplitude > 1.0 after normalization)
        samples = 24000  # 1 second
        audio = np.ones(samples) * 0.99  # Near max amplitude
        audio_int16 = (audio * 32767).astype(np.int16)

        result = validator.validate(audio_int16.tobytes())

        # Check clipping metric
        clipping_metric = next(
            (m for m in result.metrics if m.name == "clipping_ratio"),
            None,
        )
        assert clipping_metric is not None

    def test_composite_score(self):
        """Test composite score calculation."""
        from axiom_vox.verification import QualityValidator

        validator = QualityValidator()
        audio = self._generate_test_audio()

        result = validator.validate(audio)
        result.calculate_composite()

        assert result.composite_score >= 0
        assert result.composite_score <= 1.0


class TestIntelligibilityEstimator:
    """Tests for intelligibility estimator."""

    def _generate_speech_like_audio(
        self,
        duration_ms: float = 1000,
        sample_rate: int = 24000,
    ) -> bytes:
        """Generate speech-like test audio with modulation."""
        samples = int(sample_rate * duration_ms / 1000)
        t = np.arange(samples) / sample_rate

        # Base frequency with modulation (speech-like)
        modulation = 0.3 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
        audio = (0.5 + modulation) * np.sin(2 * np.pi * 200 * t)

        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def test_estimate_intelligibility(self):
        """Test intelligibility estimation."""
        from axiom_vox.verification import IntelligibilityEstimator

        estimator = IntelligibilityEstimator()
        audio = self._generate_speech_like_audio()

        score = estimator.estimate(audio, sample_rate=24000)

        assert 0 <= score <= 1


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthChecker:
    """Tests for health checker."""

    @pytest.mark.asyncio
    async def test_register_and_run_check(self):
        """Test registering and running a health check."""
        from axiom_vox.verification import (
            HealthChecker,
            HealthCheckResult,
            HealthStatus,
        )

        checker = HealthChecker()

        @checker.check(name="test_check", critical=True)
        async def check_test():
            return HealthCheckResult(
                name="test_check",
                status=HealthStatus.HEALTHY,
                message="All good",
            )

        report = await checker.check_health()

        assert len(report.checks) >= 1
        test_check = next(
            (c for c in report.checks if c.name == "test_check"),
            None,
        )
        assert test_check is not None
        assert test_check.healthy

    @pytest.mark.asyncio
    async def test_unhealthy_check(self):
        """Test unhealthy check handling."""
        from axiom_vox.verification import (
            HealthChecker,
            HealthCheckConfig,
            HealthCheckResult,
            HealthStatus,
        )

        checker = HealthChecker(HealthCheckConfig(
            include_system_metrics=False,
        ))

        @checker.check(name="failing_check", critical=True)
        async def check_failing():
            return HealthCheckResult(
                name="failing_check",
                status=HealthStatus.UNHEALTHY,
                message="Something is wrong",
            )

        report = await checker.check_health()

        assert report.overall_status == HealthStatus.UNHEALTHY
        assert not report.ready

    @pytest.mark.asyncio
    async def test_degraded_check(self):
        """Test degraded check handling."""
        from axiom_vox.verification import (
            HealthChecker,
            HealthCheckConfig,
            HealthCheckResult,
            HealthStatus,
        )

        checker = HealthChecker(HealthCheckConfig(
            include_system_metrics=False,
        ))

        @checker.check(name="healthy_check")
        async def check_healthy():
            return HealthCheckResult(
                name="healthy_check",
                status=HealthStatus.HEALTHY,
                message="Good",
            )

        @checker.check(name="degraded_check")
        async def check_degraded():
            return HealthCheckResult(
                name="degraded_check",
                status=HealthStatus.DEGRADED,
                message="Partially working",
            )

        report = await checker.check_health()

        assert report.overall_status == HealthStatus.DEGRADED
        assert report.ready  # Degraded but still operational

    @pytest.mark.asyncio
    async def test_liveness_probe(self):
        """Test liveness probe."""
        from axiom_vox.verification import LivenessProbe

        probe = LivenessProbe()
        status = await probe.check()

        assert status["alive"] is True

    @pytest.mark.asyncio
    async def test_readiness_probe(self):
        """Test readiness probe."""
        from axiom_vox.verification import (
            ReadinessProbe,
            HealthChecker,
            HealthCheckConfig,
            HealthCheckResult,
            HealthStatus,
        )

        checker = HealthChecker(HealthCheckConfig(
            include_system_metrics=False,
        ))

        @checker.check(name="ready_check", critical=True)
        async def check_ready():
            return HealthCheckResult(
                name="ready_check",
                status=HealthStatus.HEALTHY,
                message="Ready",
            )

        probe = ReadinessProbe(health_checker=checker)
        status = await probe.check()

        assert status["ready"] is True

    @pytest.mark.asyncio
    async def test_uptime(self):
        """Test uptime tracking."""
        from axiom_vox.verification import HealthChecker

        checker = HealthChecker()

        await asyncio.sleep(0.1)

        uptime = checker.get_uptime()
        assert uptime >= 0.1


# ============================================================================
# Integration Tests
# ============================================================================


class TestVerificationIntegration:
    """Integration tests for verification layer."""

    def test_imports(self):
        """Test all verification imports work."""
        from axiom_vox.verification import (
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
            # E2E testing
            E2ETestRunner,
            # Benchmarking
            BenchmarkSuite,
            # Quality
            QualityValidator,
            # Health
            HealthChecker,
        )

        assert TestStatus is not None
        assert E2ETestRunner is not None
        assert BenchmarkSuite is not None
        assert QualityValidator is not None
        assert HealthChecker is not None

    def test_main_module_exports(self):
        """Test verification exports from main module."""
        from axiom_vox import (
            TestStatus,
            TestSeverity,
            E2ETestRunner,
            BenchmarkSuite,
            QualityValidator,
            HealthChecker,
            __version__,
        )

        assert TestStatus is not None
        assert TestSeverity is not None
        assert E2ETestRunner is not None
        assert BenchmarkSuite is not None
        assert QualityValidator is not None
        assert HealthChecker is not None
        assert __version__ == "0.15.0"

    def test_verification_report(self):
        """Test VerificationReport model."""
        from axiom_vox.verification import (
            VerificationReport,
            TestSuiteResult,
            BenchmarkResult,
            BenchmarkType,
            QualityResult,
            SystemHealthReport,
            HealthStatus,
        )

        report = VerificationReport(
            test_results=TestSuiteResult(name="Tests"),
            benchmark_results=[
                BenchmarkResult(
                    name="benchmark",
                    benchmark_type=BenchmarkType.LATENCY,
                ),
            ],
            quality_results=QualityResult(),
            health_report=SystemHealthReport(
                overall_status=HealthStatus.HEALTHY,
                ready=True,
            ),
        )

        assert report.all_passed  # Empty results = all passed
        summary = report.summary()
        assert "VERIFICATION REPORT" in summary


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
