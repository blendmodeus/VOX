"""
VØX Documentation - Examples
----------------------------

Runnable code examples with validation.

AXIØM Phase 11: Document - "How do we teach this to others?"
"""

import asyncio
import io
import logging
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable

from .models import (
    Example,
    ExampleResult,
    ExampleStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class ExampleConfig:
    """
    Configuration for example runner.

    Attributes:
        timeout: Default timeout in seconds
        capture_output: Capture stdout/stderr
        validate_output: Validate against expected output
        stop_on_failure: Stop running on first failure
        parallel: Run examples in parallel
    """
    timeout: float = 30.0
    capture_output: bool = True
    validate_output: bool = True
    stop_on_failure: bool = False
    parallel: bool = False
    max_workers: int = 4


class ExampleRunner:
    """
    Runner for executable code examples.

    Features:
        - Execute Python examples
        - Capture and validate output
        - Parallel execution
        - Async support
        - Output comparison
    """

    def __init__(
        self,
        config: Optional[ExampleConfig] = None,
    ):
        """
        Initialize example runner.

        Args:
            config: Runner configuration
        """
        self.config = config or ExampleConfig()
        self._examples: List[Example] = []
        self._results: List[ExampleResult] = []

    def register(
        self,
        name: str,
        code: str,
        description: str = "",
        expected_output: Optional[str] = None,
        tags: Optional[List[str]] = None,
        requirements: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Example:
        """
        Register an example.

        Args:
            name: Example name
            code: Python code to run
            description: Example description
            expected_output: Expected output
            tags: Categorization tags
            requirements: Required modules
            timeout: Example timeout

        Returns:
            Registered example
        """
        example = Example(
            name=name,
            description=description,
            code=code,
            expected_output=expected_output,
            tags=tags or [],
            requirements=requirements or [],
            timeout=timeout or self.config.timeout,
        )
        self._examples.append(example)
        return example

    def example(
        self,
        name: Optional[str] = None,
        description: str = "",
        expected_output: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Decorator to register an example from a function.

        Args:
            name: Example name
            description: Example description
            expected_output: Expected output
            tags: Categorization tags

        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            example_name = name or func.__name__
            example_desc = description or func.__doc__ or ""

            # Get function source
            import inspect
            source = inspect.getsource(func)

            # Remove decorator and def line
            lines = source.split("\n")
            # Find the def line
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    lines = lines[i + 1:]
                    break

            # Dedent the code
            if lines:
                import textwrap
                code = textwrap.dedent("\n".join(lines))
            else:
                code = ""

            self.register(
                name=example_name,
                code=code.strip(),
                description=example_desc,
                expected_output=expected_output,
                tags=tags,
            )

            return func

        return decorator

    async def run_all(
        self,
        on_progress: Optional[Callable[[int, int, ExampleResult], None]] = None,
    ) -> List[ExampleResult]:
        """
        Run all registered examples.

        Args:
            on_progress: Progress callback

        Returns:
            List of results
        """
        results = []
        total = len(self._examples)

        if self.config.parallel:
            results = await self._run_parallel(on_progress)
        else:
            for i, example in enumerate(self._examples):
                result = await self.run_example(example)
                results.append(result)

                if on_progress:
                    on_progress(i + 1, total, result)

                if self.config.stop_on_failure and result.status == ExampleStatus.FAILED:
                    break

        self._results = results
        return results

    async def run_by_tag(
        self,
        tag: str,
        on_progress: Optional[Callable[[int, int, ExampleResult], None]] = None,
    ) -> List[ExampleResult]:
        """
        Run examples with specific tag.

        Args:
            tag: Tag to filter by
            on_progress: Progress callback

        Returns:
            Results for matching examples
        """
        examples = [e for e in self._examples if tag in e.tags]
        results = []

        for i, example in enumerate(examples):
            result = await self.run_example(example)
            results.append(result)

            if on_progress:
                on_progress(i + 1, len(examples), result)

        return results

    async def run_example(self, example: Example) -> ExampleResult:
        """
        Run a single example.

        Args:
            example: Example to run

        Returns:
            Execution result
        """
        start_time = time.time()

        # Check requirements
        for req in example.requirements:
            try:
                __import__(req)
            except ImportError:
                return ExampleResult(
                    example=example,
                    status=ExampleStatus.SKIPPED,
                    error=f"Missing requirement: {req}",
                    duration_ms=(time.time() - start_time) * 1000,
                )

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            # Build execution context
            context = {
                "__name__": "__main__",
                "__doc__": None,
            }

            # Run setup code
            if example.setup_code:
                exec(example.setup_code, context)

            # Prepare full code
            code = example.code

            # Check if code is async
            if "await " in code or "async " in code:
                # Wrap in async function
                code = f"async def __example_main__():\n" + \
                       "\n".join(f"    {line}" for line in code.split("\n")) + \
                       "\n__example_result__ = __example_main__()"
                exec(code, context)
                # Run the coroutine
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    await asyncio.wait_for(
                        context["__example_result__"],
                        timeout=example.timeout,
                    )
            else:
                # Run synchronously
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, context)

            # Run teardown code
            if example.teardown_code:
                exec(example.teardown_code, context)

            output = stdout_capture.getvalue()
            duration_ms = (time.time() - start_time) * 1000

            # Validate output
            if self.config.validate_output and example.expected_output:
                if not self._validate_output(output, example.expected_output):
                    return ExampleResult(
                        example=example,
                        status=ExampleStatus.FAILED,
                        output=output,
                        error=f"Output mismatch.\nExpected:\n{example.expected_output}\nGot:\n{output}",
                        duration_ms=duration_ms,
                    )

            return ExampleResult(
                example=example,
                status=ExampleStatus.PASSED,
                output=output,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            return ExampleResult(
                example=example,
                status=ExampleStatus.FAILED,
                error=f"Timeout after {example.timeout}s",
                duration_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ExampleResult(
                example=example,
                status=ExampleStatus.FAILED,
                output=stdout_capture.getvalue(),
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _run_parallel(
        self,
        on_progress: Optional[Callable[[int, int, ExampleResult], None]] = None,
    ) -> List[ExampleResult]:
        """Run examples in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_workers)
        results: List[ExampleResult] = [None] * len(self._examples)  # type: ignore
        completed = 0
        total = len(self._examples)
        lock = asyncio.Lock()

        async def run_with_semaphore(idx: int, example: Example):
            nonlocal completed
            async with semaphore:
                result = await self.run_example(example)
                results[idx] = result

                async with lock:
                    completed += 1
                    if on_progress:
                        on_progress(completed, total, result)

        tasks = [
            run_with_semaphore(i, example)
            for i, example in enumerate(self._examples)
        ]
        await asyncio.gather(*tasks)

        return results

    def _validate_output(self, actual: str, expected: str) -> bool:
        """Validate output against expected."""
        # Normalize whitespace
        actual = actual.strip()
        expected = expected.strip()

        # Exact match
        if actual == expected:
            return True

        # Line-by-line comparison (ignore trailing whitespace)
        actual_lines = [l.rstrip() for l in actual.split("\n")]
        expected_lines = [l.rstrip() for l in expected.split("\n")]

        if actual_lines == expected_lines:
            return True

        # Pattern matching (... for wildcards)
        if "..." in expected:
            import re
            pattern = re.escape(expected).replace(r"\.\.\.", ".*")
            if re.match(pattern, actual, re.DOTALL):
                return True

        return False

    def get_results(self) -> List[ExampleResult]:
        """Get all results."""
        return self._results

    def get_summary(self) -> Dict[str, Any]:
        """Get results summary."""
        return {
            "total": len(self._results),
            "passed": sum(1 for r in self._results if r.status == ExampleStatus.PASSED),
            "failed": sum(1 for r in self._results if r.status == ExampleStatus.FAILED),
            "skipped": sum(1 for r in self._results if r.status == ExampleStatus.SKIPPED),
            "results": [
                {
                    "name": r.example.name,
                    "status": r.status.value,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                }
                for r in self._results
            ],
        }

    def generate_report(self) -> str:
        """Generate markdown report of results."""
        lines = [
            "# Example Execution Report",
            "",
            f"**Total:** {len(self._results)}",
            f"**Passed:** {sum(1 for r in self._results if r.passed)}",
            f"**Failed:** {sum(1 for r in self._results if not r.passed)}",
            "",
            "## Results",
            "",
        ]

        for result in self._results:
            status_icon = "✓" if result.passed else "✗"
            lines.append(f"### {status_icon} {result.example.name}")
            lines.append("")
            lines.append(f"*{result.example.description}*")
            lines.append("")

            lines.append("```python")
            lines.append(result.example.code)
            lines.append("```")
            lines.append("")

            if result.output:
                lines.append("**Output:**")
                lines.append("```")
                lines.append(result.output)
                lines.append("```")
                lines.append("")

            if result.error:
                lines.append("**Error:**")
                lines.append("```")
                lines.append(result.error)
                lines.append("```")
                lines.append("")

            lines.append(f"*Duration: {result.duration_ms:.2f}ms*")
            lines.append("")

        return "\n".join(lines)


def create_vox_examples() -> ExampleRunner:
    """
    Create VØX example runner with standard examples.

    Returns:
        Configured example runner
    """
    runner = ExampleRunner()

    # ========================================================================
    # Basic Examples
    # ========================================================================

    runner.register(
        name="basic_import",
        description="Basic VØX import",
        code="""
from axiom_vox import __version__
print(f"VØX Version: {__version__}")
""",
        expected_output="VØX Version: ...",
        tags=["basic", "quick"],
    )

    runner.register(
        name="voice_space_director",
        description="Using VoiceSpaceDirector for voice matching",
        code="""
from axiom_vox import VoiceSpaceDirector

director = VoiceSpaceDirector()
result = director.direct(
    text="Welcome to our platform!",
    context={"domain": "greeting"}
)
print(f"Matched voice: {result['matched_voice_id']}")
""",
        tags=["voice", "core"],
    )

    runner.register(
        name="emotion_presets",
        description="Using emotion presets",
        code="""
from axiom_vox import list_emotion_presets, get_emotion_preset

# List available presets
presets = list_emotion_presets()
print(f"Available presets: {len(presets)}")

# Get a specific preset
joy = get_emotion_preset("joy")
print(f"Joy preset - intensity: {joy.intensity}")
""",
        tags=["emotion", "core"],
    )

    # ========================================================================
    # Synthesis Examples
    # ========================================================================

    runner.register(
        name="basic_synthesis",
        description="Basic text-to-speech synthesis",
        code="""
from axiom_vox import synthesize

# Synthesize speech (placeholder without TTS backend)
try:
    result = synthesize("Hello, world!")
    print(f"Synthesized {len(result.audio)} bytes")
except Exception as e:
    print(f"Synthesis requires TTS backend: {type(e).__name__}")
""",
        tags=["synthesis", "core"],
    )

    # ========================================================================
    # SSML Examples
    # ========================================================================

    runner.register(
        name="ssml_parsing",
        description="Parsing SSML documents",
        code="""
from axiom_vox import SSMLParser

parser = SSMLParser()
doc = parser.parse('<speak>Hello <break time="500ms"/> World</speak>')
print(f"Parsed {len(doc.children)} elements")
""",
        tags=["ssml", "parsing"],
    )

    runner.register(
        name="ssml_generation",
        description="Generating SSML documents",
        code="""
from axiom_vox import SSMLGenerator, SSMLBreak

generator = SSMLGenerator()
generator.add_text("Hello")
generator.add_element(SSMLBreak(time="500ms"))
generator.add_text("World")

ssml = generator.generate()
print(ssml[:50] + "...")
""",
        tags=["ssml", "generation"],
    )

    # ========================================================================
    # Analytics Examples
    # ========================================================================

    runner.register(
        name="analytics_collector",
        description="Using the metrics collector",
        code="""
from axiom_vox import get_collector, set_collector, VoxMetricsCollector

# Create collector
collector = VoxMetricsCollector()
set_collector(collector)

# Record a synthesis
collector.record_synthesis(
    text="Test synthesis",
    voice_id="warm",
    duration_ms=500,
    audio_bytes=48000,
)

# Get stats
stats = collector.get_stats()
print(f"Total syntheses: {stats['total_syntheses']}")
""",
        tags=["analytics", "metrics"],
    )

    # ========================================================================
    # SDK Examples
    # ========================================================================

    runner.register(
        name="sdk_client",
        description="Using the VoxClient SDK",
        code="""
from axiom_vox import VoxClient, VoxConfig, Environment

# Create client
config = VoxConfig(
    api_key="test_key",
    environment=Environment.DEVELOPMENT,
)

# Client would connect to API
print(f"Config environment: {config.environment.value}")
""",
        tags=["sdk", "client"],
    )

    # ========================================================================
    # Performance Examples
    # ========================================================================

    runner.register(
        name="caching",
        description="Using the audio cache",
        code="""
from axiom_vox import AudioCache

cache = AudioCache(max_entries=100)

# Cache audio
cache.put("Hello", "warm", b"audio_data")

# Retrieve from cache
audio = cache.get("Hello", "warm")
print(f"Cache hit: {audio is not None}")

# Check stats
stats = cache.get_stats()
print(f"Cache entries: {stats.items}")
""",
        tags=["performance", "cache"],
    )

    runner.register(
        name="batch_optimizer",
        description="Using the batch optimizer",
        code="""
import asyncio
from axiom_vox import BatchOptimizer, BatchStrategy

async def run():
    optimizer = BatchOptimizer(
        max_concurrent=5,
        strategy=BatchStrategy.PARALLEL,
    )

    items = list(range(5))

    async def processor(item):
        return item * 2

    result = await optimizer.process(items, processor)
    print(f"Processed {result.stats.completed_items} items")

asyncio.run(run())
""",
        tags=["performance", "batch"],
    )

    # ========================================================================
    # Verification Examples
    # ========================================================================

    runner.register(
        name="quality_validator",
        description="Using the quality validator",
        code="""
import numpy as np
from axiom_vox import QualityValidator

validator = QualityValidator()

# Generate test audio
audio = np.sin(np.linspace(0, 100, 24000)) * 0.5
audio_bytes = (audio * 32767).astype(np.int16).tobytes()

# Validate quality
result = validator.validate(audio_bytes)
print(f"Quality metrics: {len(result.metrics)}")
""",
        tags=["verification", "quality"],
        requirements=["numpy"],
    )

    return runner


async def run_examples() -> List[ExampleResult]:
    """
    Run standard VØX examples.

    Returns:
        Example results
    """
    runner = create_vox_examples()
    return await runner.run_all()
