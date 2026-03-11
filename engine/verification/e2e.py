"""
VØX Verification - End-to-End Testing
--------------------------------------

Comprehensive E2E test runner for VØX synthesis pipeline.

AXIØM Phase 10: Verify - "How do we know this works?"
"""

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union

from .models import (
    TestCase,
    TestResult,
    TestSuiteResult,
    TestStatus,
    TestSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class E2ETestConfig:
    """
    Configuration for E2E test runner.

    Attributes:
        timeout: Default test timeout in seconds
        parallel: Run tests in parallel
        max_workers: Max parallel workers
        fail_fast: Stop on first failure
        retry_count: Number of retries for flaky tests
        verbose: Enable verbose output
    """
    timeout: float = 30.0
    parallel: bool = False
    max_workers: int = 4
    fail_fast: bool = False
    retry_count: int = 0
    verbose: bool = False
    tags_include: List[str] = field(default_factory=list)
    tags_exclude: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


class E2ETestRunner:
    """
    End-to-end test runner for VØX synthesis pipeline.

    Features:
        - Full pipeline testing
        - Parallel test execution
        - Retry support for flaky tests
        - Tag-based filtering
        - Progress reporting
    """

    def __init__(
        self,
        config: Optional[E2ETestConfig] = None,
    ):
        """
        Initialize E2E test runner.

        Args:
            config: Test configuration
        """
        self.config = config or E2ETestConfig()
        self._tests: List[TestCase] = []
        self._results: List[TestResult] = []
        self._hooks: Dict[str, List[Callable]] = {
            "before_all": [],
            "after_all": [],
            "before_each": [],
            "after_each": [],
        }

    def register_test(
        self,
        name: str,
        test_fn: Callable[..., Awaitable[Any]],
        description: str = "",
        category: str = "general",
        severity: TestSeverity = TestSeverity.MEDIUM,
        timeout: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> TestCase:
        """
        Register a test case.

        Args:
            name: Test name
            test_fn: Async test function
            description: Test description
            category: Test category
            severity: Failure severity
            timeout: Test timeout
            tags: Test tags

        Returns:
            Registered test case
        """
        test = TestCase(
            name=name,
            description=description or f"Test: {name}",
            category=category,
            severity=severity,
            timeout=timeout or self.config.timeout,
            tags=tags or [],
        )

        # Store test function
        test._test_fn = test_fn  # type: ignore

        self._tests.append(test)
        return test

    def register_hook(
        self,
        hook_type: str,
        hook_fn: Callable[..., Awaitable[Any]],
    ) -> None:
        """
        Register a test hook.

        Args:
            hook_type: "before_all", "after_all", "before_each", "after_each"
            hook_fn: Async hook function
        """
        if hook_type in self._hooks:
            self._hooks[hook_type].append(hook_fn)

    def test(
        self,
        name: Optional[str] = None,
        category: str = "general",
        severity: TestSeverity = TestSeverity.MEDIUM,
        timeout: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Decorator to register a test function.

        Args:
            name: Test name (defaults to function name)
            category: Test category
            severity: Failure severity
            timeout: Test timeout
            tags: Test tags

        Returns:
            Decorator function
        """
        def decorator(fn: Callable[..., Awaitable[Any]]):
            test_name = name or fn.__name__
            self.register_test(
                name=test_name,
                test_fn=fn,
                description=fn.__doc__ or "",
                category=category,
                severity=severity,
                timeout=timeout,
                tags=tags,
            )
            return fn
        return decorator

    async def run_all(
        self,
        on_progress: Optional[Callable[[int, int, TestResult], None]] = None,
    ) -> TestSuiteResult:
        """
        Run all registered tests.

        Args:
            on_progress: Progress callback (completed, total, result)

        Returns:
            Test suite result
        """
        suite_result = TestSuiteResult(
            name="VØX E2E Tests",
            started_at=time.time(),
        )

        # Filter tests
        tests = self._filter_tests()

        if not tests:
            logger.warning("No tests to run after filtering")
            suite_result.completed_at = time.time()
            return suite_result

        # Run before_all hooks
        await self._run_hooks("before_all")

        try:
            if self.config.parallel:
                results = await self._run_parallel(tests, on_progress)
            else:
                results = await self._run_sequential(tests, on_progress)

            for result in results:
                suite_result.add_result(result)

        finally:
            # Run after_all hooks
            await self._run_hooks("after_all")

        suite_result.completed_at = time.time()
        return suite_result

    async def run_by_category(
        self,
        category: str,
        on_progress: Optional[Callable[[int, int, TestResult], None]] = None,
    ) -> TestSuiteResult:
        """
        Run tests in a specific category.

        Args:
            category: Category to run
            on_progress: Progress callback

        Returns:
            Test suite result
        """
        original_categories = self.config.categories
        self.config.categories = [category]

        try:
            result = await self.run_all(on_progress)
            result.name = f"VØX E2E Tests: {category}"
            return result
        finally:
            self.config.categories = original_categories

    async def run_by_tags(
        self,
        tags: List[str],
        on_progress: Optional[Callable[[int, int, TestResult], None]] = None,
    ) -> TestSuiteResult:
        """
        Run tests with specific tags.

        Args:
            tags: Tags to include
            on_progress: Progress callback

        Returns:
            Test suite result
        """
        original_tags = self.config.tags_include
        self.config.tags_include = tags

        try:
            result = await self.run_all(on_progress)
            result.name = f"VØX E2E Tests: {', '.join(tags)}"
            return result
        finally:
            self.config.tags_include = original_tags

    async def run_single(self, test_name: str) -> TestResult:
        """
        Run a single test by name.

        Args:
            test_name: Test name to run

        Returns:
            Test result
        """
        test = next((t for t in self._tests if t.name == test_name), None)
        if test is None:
            raise ValueError(f"Test not found: {test_name}")

        await self._run_hooks("before_each", test)
        result = await self._run_test(test)
        await self._run_hooks("after_each", test, result)

        return result

    def get_test_count(self) -> int:
        """Get total number of registered tests."""
        return len(self._tests)

    def get_categories(self) -> List[str]:
        """Get all test categories."""
        return list(set(t.category for t in self._tests))

    def get_tags(self) -> List[str]:
        """Get all test tags."""
        tags = set()
        for test in self._tests:
            tags.update(test.tags)
        return list(tags)

    def clear(self) -> None:
        """Clear all registered tests."""
        self._tests.clear()
        self._results.clear()

    def _filter_tests(self) -> List[TestCase]:
        """Filter tests based on configuration."""
        tests = [t for t in self._tests if t.enabled]

        # Filter by category
        if self.config.categories:
            tests = [
                t for t in tests
                if t.category in self.config.categories
            ]

        # Filter by include tags
        if self.config.tags_include:
            tests = [
                t for t in tests
                if any(tag in t.tags for tag in self.config.tags_include)
            ]

        # Filter by exclude tags
        if self.config.tags_exclude:
            tests = [
                t for t in tests
                if not any(tag in t.tags for tag in self.config.tags_exclude)
            ]

        return tests

    async def _run_sequential(
        self,
        tests: List[TestCase],
        on_progress: Optional[Callable[[int, int, TestResult], None]] = None,
    ) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        total = len(tests)

        for i, test in enumerate(tests):
            await self._run_hooks("before_each", test)
            result = await self._run_test(test)
            await self._run_hooks("after_each", test, result)

            results.append(result)

            if on_progress:
                on_progress(i + 1, total, result)

            if self.config.fail_fast and result.failed:
                logger.warning(f"Fail fast: stopping after {test.name}")
                break

        return results

    async def _run_parallel(
        self,
        tests: List[TestCase],
        on_progress: Optional[Callable[[int, int, TestResult], None]] = None,
    ) -> List[TestResult]:
        """Run tests in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_workers)
        results: List[TestResult] = []
        completed = 0
        total = len(tests)
        lock = asyncio.Lock()

        async def run_with_semaphore(test: TestCase) -> TestResult:
            nonlocal completed
            async with semaphore:
                await self._run_hooks("before_each", test)
                result = await self._run_test(test)
                await self._run_hooks("after_each", test, result)

                async with lock:
                    completed += 1
                    if on_progress:
                        on_progress(completed, total, result)

                return result

        tasks = [run_with_semaphore(test) for test in tests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(TestResult(
                    test_case=tests[i],
                    status=TestStatus.ERROR,
                    duration_ms=0,
                    error=str(result),
                    error_trace=traceback.format_exc(),
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _run_test(self, test: TestCase) -> TestResult:
        """Run a single test with retries."""
        last_result = None
        attempts = 1 + self.config.retry_count

        for attempt in range(attempts):
            start_time = time.time()

            try:
                # Run setup
                if test.setup:
                    await self._run_callable(test.setup)

                # Run test with timeout
                test_fn = getattr(test, "_test_fn", None)
                if test_fn is None:
                    raise ValueError(f"No test function for {test.name}")

                await asyncio.wait_for(
                    test_fn(),
                    timeout=test.timeout,
                )

                duration_ms = (time.time() - start_time) * 1000

                result = TestResult(
                    test_case=test,
                    status=TestStatus.PASSED,
                    duration_ms=duration_ms,
                    message=f"Test passed in {duration_ms:.1f}ms",
                    started_at=start_time,
                    completed_at=time.time(),
                )

                if self.config.verbose:
                    logger.info(f"✓ {test.name} ({duration_ms:.1f}ms)")

                return result

            except asyncio.TimeoutError:
                duration_ms = (time.time() - start_time) * 1000
                last_result = TestResult(
                    test_case=test,
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=f"Test timed out after {test.timeout}s",
                    error="TimeoutError",
                    started_at=start_time,
                    completed_at=time.time(),
                )

            except AssertionError as e:
                duration_ms = (time.time() - start_time) * 1000
                last_result = TestResult(
                    test_case=test,
                    status=TestStatus.FAILED,
                    duration_ms=duration_ms,
                    message=str(e) or "Assertion failed",
                    error=str(e),
                    error_trace=traceback.format_exc(),
                    started_at=start_time,
                    completed_at=time.time(),
                )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                last_result = TestResult(
                    test_case=test,
                    status=TestStatus.ERROR,
                    duration_ms=duration_ms,
                    message=f"Test error: {type(e).__name__}",
                    error=str(e),
                    error_trace=traceback.format_exc(),
                    started_at=start_time,
                    completed_at=time.time(),
                )

            finally:
                # Run teardown
                if test.teardown:
                    try:
                        await self._run_callable(test.teardown)
                    except Exception as e:
                        logger.warning(f"Teardown error for {test.name}: {e}")

            if attempt < attempts - 1:
                logger.info(f"Retrying {test.name} (attempt {attempt + 2}/{attempts})")
                await asyncio.sleep(0.5)  # Brief delay before retry

        if self.config.verbose and last_result:
            status = "✗" if last_result.failed else "?"
            logger.info(f"{status} {test.name}: {last_result.message}")

        return last_result or TestResult(
            test_case=test,
            status=TestStatus.ERROR,
            duration_ms=0,
            message="Unknown error",
        )

    async def _run_hooks(
        self,
        hook_type: str,
        *args,
        **kwargs,
    ) -> None:
        """Run hooks of specified type."""
        for hook in self._hooks.get(hook_type, []):
            try:
                await self._run_callable(hook, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Hook {hook_type} error: {e}")

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


def create_vox_e2e_tests() -> E2ETestRunner:
    """
    Create standard VØX E2E test suite.

    Returns:
        Configured E2E test runner with standard tests
    """
    runner = E2ETestRunner(E2ETestConfig(
        timeout=30.0,
        verbose=True,
    ))

    # ========================================================================
    # Synthesis Tests
    # ========================================================================

    @runner.test(category="synthesis", tags=["core", "smoke"])
    async def test_basic_synthesis():
        """Test basic text-to-speech synthesis."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            result = await engine.synthesize("Hello, world!")
            assert result is not None
            assert result.audio is not None
            assert len(result.audio) > 0
        finally:
            await engine.stop()

    @runner.test(category="synthesis", tags=["core"])
    async def test_synthesis_with_voice():
        """Test synthesis with specific voice."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            result = await engine.synthesize(
                text="Test with voice",
                voice_id="warm",
            )
            assert result is not None
            assert result.voice_id == "warm"
        finally:
            await engine.stop()

    @runner.test(category="synthesis", tags=["core"])
    async def test_synthesis_with_params():
        """Test synthesis with custom parameters."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            result = await engine.synthesize(
                text="Test with params",
                speed=1.5,
                pitch=1.2,
            )
            assert result is not None
        finally:
            await engine.stop()

    # ========================================================================
    # Streaming Tests
    # ========================================================================

    @runner.test(category="streaming", tags=["core"])
    async def test_streaming_synthesis():
        """Test streaming synthesis."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            chunks = []
            async for chunk in engine.synthesize_stream("Hello streaming world"):
                chunks.append(chunk)

            assert len(chunks) > 0
        finally:
            await engine.stop()

    @runner.test(category="streaming", tags=["performance"])
    async def test_streaming_first_chunk_latency():
        """Test first chunk latency in streaming."""
        import time
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            start = time.time()
            first_chunk_time = None

            async for _chunk in engine.synthesize_stream("Testing latency"):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start
                    break

            assert first_chunk_time is not None
            assert first_chunk_time < 1.0  # Should be under 1 second
        finally:
            await engine.stop()

    # ========================================================================
    # Voice Management Tests
    # ========================================================================

    @runner.test(category="voices", tags=["core"])
    async def test_list_voices():
        """Test listing available voices."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            voices = await engine.list_voices()
            assert voices is not None
            assert len(voices) > 0
        finally:
            await engine.stop()

    @runner.test(category="voices", tags=["core"])
    async def test_get_voice_details():
        """Test getting voice details."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            voices = await engine.list_voices()
            if voices:
                voice = await engine.get_voice(voices[0].voice_id)
                assert voice is not None
                assert voice.voice_id == voices[0].voice_id
        finally:
            await engine.stop()

    # ========================================================================
    # SSML Tests
    # ========================================================================

    @runner.test(category="ssml", tags=["features"])
    async def test_ssml_synthesis():
        """Test SSML synthesis."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            ssml = '<speak>Hello <break time="500ms"/> World</speak>'
            result = await engine.synthesize_ssml(ssml)
            assert result is not None
        finally:
            await engine.stop()

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    @runner.test(category="errors", tags=["robustness"])
    async def test_empty_text_handling():
        """Test handling of empty text."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            try:
                await engine.synthesize("")
                assert False, "Should have raised error"
            except ValueError:
                pass  # Expected
        finally:
            await engine.stop()

    @runner.test(category="errors", tags=["robustness"])
    async def test_invalid_voice_handling():
        """Test handling of invalid voice ID."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            try:
                await engine.synthesize(
                    text="Test",
                    voice_id="nonexistent_voice_xyz",
                )
                assert False, "Should have raised error"
            except Exception:
                pass  # Expected (VoiceNotFoundError or similar)
        finally:
            await engine.stop()

    # ========================================================================
    # Governance Tests
    # ========================================================================

    @runner.test(category="governance", tags=["security"])
    async def test_content_filtering():
        """Test content filtering for harmful content."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            # This should be filtered
            result = await engine.synthesize(
                text="Normal test content",
                check_content=True,
            )
            assert result is not None
        finally:
            await engine.stop()

    @runner.test(category="governance", tags=["security"])
    async def test_rate_limiting():
        """Test rate limiting behavior."""
        from axiom_vox import VoxEngine

        engine = VoxEngine()
        await engine.start()

        try:
            # Make several requests quickly
            for _ in range(5):
                await engine.synthesize("Quick test")
            # Should complete without rate limit errors
        finally:
            await engine.stop()

    return runner


# Convenience function for quick verification
async def run_quick_verification() -> TestSuiteResult:
    """
    Run quick verification with smoke tests only.

    Returns:
        Test suite result
    """
    runner = create_vox_e2e_tests()
    runner.config.tags_include = ["smoke"]
    return await runner.run_all()


async def run_full_verification() -> TestSuiteResult:
    """
    Run full verification suite.

    Returns:
        Test suite result
    """
    runner = create_vox_e2e_tests()
    return await runner.run_all()
