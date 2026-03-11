"""
VØX Verification - Benchmarking
-------------------------------

Performance benchmarking suite for VØX.

AXIØM Phase 10: Verify - "How do we know this works?"
"""

import asyncio
import gc
import logging
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Awaitable

from .models import (
    BenchmarkResult,
    BenchmarkType,
    BenchmarkMetric,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmarks.

    Attributes:
        iterations: Number of benchmark iterations
        warmup_iterations: Warmup iterations (not counted)
        cooldown_seconds: Delay between iterations
        memory_tracking: Track memory usage
        gc_before_run: Run GC before each iteration
    """
    iterations: int = 10
    warmup_iterations: int = 2
    cooldown_seconds: float = 0.1
    memory_tracking: bool = True
    gc_before_run: bool = True


@dataclass
class BenchmarkDefinition:
    """
    Definition of a benchmark.

    Attributes:
        name: Benchmark name
        benchmark_type: Type of benchmark
        fn: Benchmark function
        setup: Optional setup function
        teardown: Optional teardown function
        thresholds: Metric thresholds
    """
    name: str
    benchmark_type: BenchmarkType
    fn: Callable[..., Awaitable[Any]]
    setup: Optional[Callable[..., Awaitable[Any]]] = None
    teardown: Optional[Callable[..., Awaitable[Any]]] = None
    thresholds: Dict[str, float] = field(default_factory=dict)
    description: str = ""


class BenchmarkSuite:
    """
    Performance benchmark suite for VØX.

    Features:
        - Latency benchmarks
        - Throughput benchmarks
        - Memory usage tracking
        - Statistical analysis
        - Threshold validation
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
    ):
        """
        Initialize benchmark suite.

        Args:
            config: Benchmark configuration
        """
        self.config = config or BenchmarkConfig()
        self._benchmarks: List[BenchmarkDefinition] = []
        self._results: List[BenchmarkResult] = []

    def register(
        self,
        name: str,
        benchmark_type: BenchmarkType,
        fn: Callable[..., Awaitable[Any]],
        setup: Optional[Callable[..., Awaitable[Any]]] = None,
        teardown: Optional[Callable[..., Awaitable[Any]]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        description: str = "",
    ) -> BenchmarkDefinition:
        """
        Register a benchmark.

        Args:
            name: Benchmark name
            benchmark_type: Type of benchmark
            fn: Benchmark function
            setup: Setup function
            teardown: Teardown function
            thresholds: Metric thresholds
            description: Description

        Returns:
            Benchmark definition
        """
        benchmark = BenchmarkDefinition(
            name=name,
            benchmark_type=benchmark_type,
            fn=fn,
            setup=setup,
            teardown=teardown,
            thresholds=thresholds or {},
            description=description,
        )
        self._benchmarks.append(benchmark)
        return benchmark

    def benchmark(
        self,
        name: Optional[str] = None,
        benchmark_type: BenchmarkType = BenchmarkType.LATENCY,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Decorator to register a benchmark.

        Args:
            name: Benchmark name
            benchmark_type: Type of benchmark
            thresholds: Metric thresholds

        Returns:
            Decorator function
        """
        def decorator(fn: Callable[..., Awaitable[Any]]):
            bench_name = name or fn.__name__
            self.register(
                name=bench_name,
                benchmark_type=benchmark_type,
                fn=fn,
                thresholds=thresholds,
                description=fn.__doc__ or "",
            )
            return fn
        return decorator

    async def run_all(
        self,
        on_progress: Optional[Callable[[int, int, BenchmarkResult], None]] = None,
    ) -> List[BenchmarkResult]:
        """
        Run all registered benchmarks.

        Args:
            on_progress: Progress callback

        Returns:
            List of benchmark results
        """
        results = []
        total = len(self._benchmarks)

        for i, benchmark in enumerate(self._benchmarks):
            logger.info(f"Running benchmark: {benchmark.name}")

            result = await self._run_benchmark(benchmark)
            results.append(result)

            if on_progress:
                on_progress(i + 1, total, result)

        self._results = results
        return results

    async def run_by_type(
        self,
        benchmark_type: BenchmarkType,
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks of specific type.

        Args:
            benchmark_type: Type to run

        Returns:
            Benchmark results
        """
        benchmarks = [
            b for b in self._benchmarks
            if b.benchmark_type == benchmark_type
        ]

        results = []
        for benchmark in benchmarks:
            result = await self._run_benchmark(benchmark)
            results.append(result)

        return results

    async def run_single(self, name: str) -> BenchmarkResult:
        """
        Run a single benchmark by name.

        Args:
            name: Benchmark name

        Returns:
            Benchmark result
        """
        benchmark = next(
            (b for b in self._benchmarks if b.name == name),
            None,
        )
        if benchmark is None:
            raise ValueError(f"Benchmark not found: {name}")

        return await self._run_benchmark(benchmark)

    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results."""
        return self._results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all results."""
        return {
            "total_benchmarks": len(self._results),
            "passed": sum(1 for r in self._results if r.passed),
            "failed": sum(1 for r in self._results if not r.passed),
            "results": [
                {
                    "name": r.name,
                    "type": r.benchmark_type.value,
                    "passed": r.passed,
                    "metrics": {
                        m.name: m.value
                        for m in r.metrics
                    },
                }
                for r in self._results
            ],
        }

    async def _run_benchmark(
        self,
        benchmark: BenchmarkDefinition,
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        result = BenchmarkResult(
            name=benchmark.name,
            benchmark_type=benchmark.benchmark_type,
            iterations=self.config.iterations,
            warmup_iterations=self.config.warmup_iterations,
            started_at=time.time(),
        )

        try:
            # Setup
            if benchmark.setup:
                await self._run_callable(benchmark.setup)

            # Warmup iterations
            for _ in range(self.config.warmup_iterations):
                if self.config.gc_before_run:
                    gc.collect()
                await benchmark.fn()
                await asyncio.sleep(self.config.cooldown_seconds)

            # Benchmark iterations
            latencies = []
            memory_samples = []

            for _ in range(self.config.iterations):
                if self.config.gc_before_run:
                    gc.collect()

                # Start memory tracking
                if self.config.memory_tracking:
                    tracemalloc.start()

                start_time = time.perf_counter()
                await benchmark.fn()
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                # Capture memory
                if self.config.memory_tracking:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    memory_samples.append(peak)

                await asyncio.sleep(self.config.cooldown_seconds)

            # Calculate metrics
            self._add_latency_metrics(result, latencies, benchmark.thresholds)

            if memory_samples:
                self._add_memory_metrics(result, memory_samples, benchmark.thresholds)

            # Calculate throughput
            if benchmark.benchmark_type == BenchmarkType.THROUGHPUT:
                self._add_throughput_metrics(result, latencies, benchmark.thresholds)

        except Exception as e:
            logger.error(f"Benchmark {benchmark.name} failed: {e}")
            result.metadata["error"] = str(e)

        finally:
            # Teardown
            if benchmark.teardown:
                try:
                    await self._run_callable(benchmark.teardown)
                except Exception as e:
                    logger.warning(f"Teardown error: {e}")

        result.completed_at = time.time()
        result.duration_ms = (result.completed_at - result.started_at) * 1000

        return result

    def _add_latency_metrics(
        self,
        result: BenchmarkResult,
        latencies: List[float],
        thresholds: Dict[str, float],
    ) -> None:
        """Add latency metrics to result."""
        if not latencies:
            return

        # Mean latency
        result.add_metric(
            name="latency_mean_ms",
            value=statistics.mean(latencies),
            unit="ms",
            threshold=thresholds.get("latency_mean_ms"),
            threshold_type="max",
        )

        # Median latency
        result.add_metric(
            name="latency_median_ms",
            value=statistics.median(latencies),
            unit="ms",
            threshold=thresholds.get("latency_median_ms"),
            threshold_type="max",
        )

        # Min/Max latency
        result.add_metric(
            name="latency_min_ms",
            value=min(latencies),
            unit="ms",
        )

        result.add_metric(
            name="latency_max_ms",
            value=max(latencies),
            unit="ms",
            threshold=thresholds.get("latency_max_ms"),
            threshold_type="max",
        )

        # Standard deviation
        if len(latencies) > 1:
            result.add_metric(
                name="latency_std_ms",
                value=statistics.stdev(latencies),
                unit="ms",
            )

        # Percentiles
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        p50_idx = int(n * 0.50)
        p90_idx = int(n * 0.90)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        result.add_metric(
            name="latency_p50_ms",
            value=sorted_latencies[min(p50_idx, n - 1)],
            unit="ms",
        )

        result.add_metric(
            name="latency_p90_ms",
            value=sorted_latencies[min(p90_idx, n - 1)],
            unit="ms",
            threshold=thresholds.get("latency_p90_ms"),
            threshold_type="max",
        )

        result.add_metric(
            name="latency_p95_ms",
            value=sorted_latencies[min(p95_idx, n - 1)],
            unit="ms",
            threshold=thresholds.get("latency_p95_ms"),
            threshold_type="max",
        )

        result.add_metric(
            name="latency_p99_ms",
            value=sorted_latencies[min(p99_idx, n - 1)],
            unit="ms",
            threshold=thresholds.get("latency_p99_ms"),
            threshold_type="max",
        )

    def _add_memory_metrics(
        self,
        result: BenchmarkResult,
        memory_samples: List[int],
        thresholds: Dict[str, float],
    ) -> None:
        """Add memory metrics to result."""
        if not memory_samples:
            return

        # Convert to MB
        memory_mb = [m / (1024 * 1024) for m in memory_samples]

        result.add_metric(
            name="memory_mean_mb",
            value=statistics.mean(memory_mb),
            unit="MB",
            threshold=thresholds.get("memory_mean_mb"),
            threshold_type="max",
        )

        result.add_metric(
            name="memory_peak_mb",
            value=max(memory_mb),
            unit="MB",
            threshold=thresholds.get("memory_peak_mb"),
            threshold_type="max",
        )

        result.add_metric(
            name="memory_min_mb",
            value=min(memory_mb),
            unit="MB",
        )

    def _add_throughput_metrics(
        self,
        result: BenchmarkResult,
        latencies: List[float],
        thresholds: Dict[str, float],
    ) -> None:
        """Add throughput metrics to result."""
        if not latencies:
            return

        # Requests per second
        mean_latency_sec = statistics.mean(latencies) / 1000
        rps = 1.0 / mean_latency_sec if mean_latency_sec > 0 else 0

        result.add_metric(
            name="throughput_rps",
            value=rps,
            unit="req/s",
            threshold=thresholds.get("throughput_rps"),
            threshold_type="min",
        )

    async def _run_callable(
        self,
        fn: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Run a callable, handling sync/async."""
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result


def create_vox_benchmarks() -> BenchmarkSuite:
    """
    Create standard VØX benchmark suite.

    Returns:
        Configured benchmark suite
    """
    suite = BenchmarkSuite(BenchmarkConfig(
        iterations=10,
        warmup_iterations=2,
        memory_tracking=True,
    ))

    # ========================================================================
    # Latency Benchmarks
    # ========================================================================

    @suite.benchmark(
        name="synthesis_latency",
        benchmark_type=BenchmarkType.LATENCY,
        thresholds={
            "latency_p95_ms": 500.0,  # 500ms p95
            "latency_max_ms": 1000.0,  # 1s max
        },
    )
    async def benchmark_synthesis_latency():
        """Measure basic synthesis latency."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()
        try:
            await engine.synthesize("Hello, this is a benchmark test.")
        finally:
            await engine.stop()

    @suite.benchmark(
        name="streaming_first_chunk",
        benchmark_type=BenchmarkType.LATENCY,
        thresholds={
            "latency_p95_ms": 200.0,  # 200ms to first chunk
        },
    )
    async def benchmark_streaming_first_chunk():
        """Measure time to first chunk in streaming."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()
        try:
            async for _chunk in engine.synthesize_stream("Streaming benchmark"):
                break  # Only measure first chunk
        finally:
            await engine.stop()

    @suite.benchmark(
        name="voice_switch_latency",
        benchmark_type=BenchmarkType.LATENCY,
        thresholds={
            "latency_mean_ms": 100.0,
        },
    )
    async def benchmark_voice_switch():
        """Measure voice switching latency."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()
        try:
            voices = await engine.list_voices()
            if len(voices) >= 2:
                await engine.synthesize("Test one", voice_id=voices[0].voice_id)
                await engine.synthesize("Test two", voice_id=voices[1].voice_id)
        finally:
            await engine.stop()

    # ========================================================================
    # Throughput Benchmarks
    # ========================================================================

    @suite.benchmark(
        name="synthesis_throughput",
        benchmark_type=BenchmarkType.THROUGHPUT,
        thresholds={
            "throughput_rps": 1.0,  # At least 1 request per second
        },
    )
    async def benchmark_synthesis_throughput():
        """Measure synthesis throughput."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()
        try:
            # Run multiple syntheses in parallel
            tasks = [
                engine.synthesize(f"Throughput test {i}")
                for i in range(5)
            ]
            await asyncio.gather(*tasks)
        finally:
            await engine.stop()

    # ========================================================================
    # Memory Benchmarks
    # ========================================================================

    @suite.benchmark(
        name="memory_synthesis",
        benchmark_type=BenchmarkType.MEMORY,
        thresholds={
            "memory_peak_mb": 500.0,  # 500MB peak
        },
    )
    async def benchmark_memory_synthesis():
        """Measure memory usage during synthesis."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()
        try:
            # Synthesize longer text
            text = "This is a longer text for memory benchmarking. " * 10
            await engine.synthesize(text)
        finally:
            await engine.stop()

    @suite.benchmark(
        name="memory_streaming",
        benchmark_type=BenchmarkType.MEMORY,
        thresholds={
            "memory_peak_mb": 200.0,  # Lower for streaming
        },
    )
    async def benchmark_memory_streaming():
        """Measure memory usage during streaming."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()
        try:
            text = "Streaming memory test. " * 10
            async for _chunk in engine.synthesize_stream(text):
                pass
        finally:
            await engine.stop()

    return suite


async def run_benchmarks() -> List[BenchmarkResult]:
    """
    Run standard VØX benchmarks.

    Returns:
        Benchmark results
    """
    suite = create_vox_benchmarks()
    return await suite.run_all()


async def run_latency_benchmarks() -> List[BenchmarkResult]:
    """
    Run latency benchmarks only.

    Returns:
        Benchmark results
    """
    suite = create_vox_benchmarks()
    return await suite.run_by_type(BenchmarkType.LATENCY)
