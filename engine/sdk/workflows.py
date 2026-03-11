"""
VØX SDK Workflows
-----------------

High-level workflow helpers for common VØX operations.

Features:
    - Composite operations (enroll + verify, synthesize + quality check)
    - Batch processing
    - Pipeline builders
    - Convenience methods

AXIØM Phase 8: Integrate - "How do the parts connect?"
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, AsyncIterator, Union
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    operation: str
    params: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    on_success: Optional[str] = None  # Next step name
    on_failure: Optional[str] = None  # Error handler step


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    workflow_name: str
    status: WorkflowStatus
    steps_completed: List[str] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    final_output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class SynthesisResult:
    """Result of a synthesis operation."""
    audio: bytes
    voice_id: str
    text: str
    duration_seconds: float = 0.0
    quality_score: Optional[float] = None
    sample_rate: int = 24000
    format: str = "mp3"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrollmentResult:
    """Result of an enrollment operation."""
    voice_id: str
    template_id: str
    samples_used: int
    confidence: float
    enrolled: bool = True
    warnings: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Result of a verification operation."""
    voice_id: str
    verified: bool
    similarity: float
    threshold: float
    liveness_passed: bool = True
    liveness_score: Optional[float] = None


@dataclass
class DialogueLine:
    """A line in a dialogue."""
    text: str
    character: Optional[str] = None
    voice_id: Optional[str] = None
    emotion: Optional[str] = None
    pause_after_ms: int = 0


@dataclass
class DialogueResult:
    """Result of dialogue synthesis."""
    audio: bytes
    lines: List[DialogueLine]
    voices_used: List[str]
    total_duration_seconds: float
    segments: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================================
# Synthesis Workflows
# ============================================================================


async def synthesize_with_quality_check(
    client,
    text: str,
    voice_id: Optional[str] = None,
    min_quality: float = 0.6,
    max_retries: int = 2,
    **kwargs,
) -> SynthesisResult:
    """
    Synthesize audio with automatic quality checking and retry.

    If quality is below threshold, retries with adjusted parameters.

    Args:
        client: VoxClient instance
        text: Text to synthesize
        voice_id: Voice ID to use
        min_quality: Minimum quality threshold
        max_retries: Maximum retry attempts for quality
        **kwargs: Additional synthesis parameters

    Returns:
        SynthesisResult
    """
    attempts = 0
    best_result = None
    best_quality = 0.0

    while attempts <= max_retries:
        result = await client.synthesize(text, voice_id=voice_id, **kwargs)

        quality = result.quality_score or 0.0

        if quality >= min_quality:
            return result

        if quality > best_quality:
            best_quality = quality
            best_result = result

        attempts += 1

        if attempts <= max_retries:
            # Adjust parameters for retry
            kwargs["speaking_rate"] = kwargs.get("speaking_rate", 1.0) * 0.95
            logger.info(
                f"Quality {quality:.3f} below threshold {min_quality}, "
                f"retrying ({attempts}/{max_retries})"
            )

    # Return best result if threshold not met
    logger.warning(
        f"Quality threshold not met after {max_retries} retries. "
        f"Best quality: {best_quality:.3f}"
    )
    return best_result or result


async def synthesize_verified(
    client,
    text: str,
    voice_id: str,
    speaker_audio: bytes,
    require_verification: bool = True,
    **kwargs,
) -> SynthesisResult:
    """
    Synthesize audio only if speaker is verified.

    Args:
        client: VoxClient instance
        text: Text to synthesize
        voice_id: Voice ID to use
        speaker_audio: Audio sample for speaker verification
        require_verification: Fail if verification fails
        **kwargs: Additional synthesis parameters

    Returns:
        SynthesisResult

    Raises:
        VerificationError: If verification fails and required
    """
    # First verify the speaker
    verification = await client.verify_speaker(voice_id, speaker_audio)

    if not verification.verified:
        if require_verification:
            from .errors import VerificationError
            raise VerificationError(
                voice_id=voice_id,
                similarity=verification.similarity,
                threshold=verification.threshold,
            )
        logger.warning(f"Speaker verification failed but continuing")

    # Synthesize with verified context
    kwargs["verified_speaker"] = verification.verified
    kwargs["speaker_similarity"] = verification.similarity

    return await client.synthesize(text, voice_id=voice_id, **kwargs)


async def synthesize_batch(
    client,
    items: List[Dict[str, Any]],
    max_concurrent: int = 5,
    on_complete: Optional[Callable[[int, SynthesisResult], None]] = None,
) -> List[SynthesisResult]:
    """
    Synthesize multiple items in parallel.

    Args:
        client: VoxClient instance
        items: List of synthesis parameters (each dict has text, voice_id, etc.)
        max_concurrent: Maximum concurrent requests
        on_complete: Callback when each item completes

    Returns:
        List of SynthesisResult in same order as input
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results: List[Optional[SynthesisResult]] = [None] * len(items)

    async def process_item(index: int, item: Dict[str, Any]):
        async with semaphore:
            text = item.pop("text")
            result = await client.synthesize(text, **item)
            results[index] = result
            if on_complete:
                on_complete(index, result)

    tasks = [process_item(i, item) for i, item in enumerate(items)]
    await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if r is not None]


# ============================================================================
# Biometric Workflows
# ============================================================================


async def enroll_and_verify(
    client,
    voice_id: str,
    enrollment_samples: List[bytes],
    verification_sample: bytes,
    owner_id: str,
    **kwargs,
) -> tuple[EnrollmentResult, VerificationResult]:
    """
    Enroll a voice and immediately verify.

    Useful for testing enrollment quality.

    Args:
        client: VoxClient instance
        voice_id: Voice ID to enroll
        enrollment_samples: Audio samples for enrollment
        verification_sample: Audio sample to verify against enrollment
        owner_id: Owner identifier
        **kwargs: Additional parameters

    Returns:
        Tuple of (EnrollmentResult, VerificationResult)
    """
    # Enroll
    enrollment = await client.enroll_voice(
        voice_id=voice_id,
        audio_samples=enrollment_samples,
        owner_id=owner_id,
        **kwargs,
    )

    # Verify
    verification = await client.verify_speaker(
        voice_id=voice_id,
        audio=verification_sample,
    )

    return enrollment, verification


async def continuous_verification(
    client,
    voice_id: str,
    audio_stream: AsyncIterator[bytes],
    interval_seconds: float = 5.0,
    on_verification: Optional[Callable[[VerificationResult], None]] = None,
) -> AsyncIterator[VerificationResult]:
    """
    Continuously verify speaker identity from audio stream.

    Args:
        client: VoxClient instance
        voice_id: Voice ID to verify against
        audio_stream: Async iterator of audio chunks
        interval_seconds: Seconds of audio to collect before verification
        on_verification: Callback for each verification result

    Yields:
        VerificationResult for each verification
    """
    buffer = bytearray()
    bytes_per_second = 24000 * 2  # 24kHz, 16-bit
    buffer_target = int(interval_seconds * bytes_per_second)

    async for chunk in audio_stream:
        buffer.extend(chunk)

        if len(buffer) >= buffer_target:
            # Verify accumulated audio
            result = await client.verify_speaker(
                voice_id=voice_id,
                audio=bytes(buffer),
            )

            if on_verification:
                on_verification(result)

            yield result
            buffer.clear()

    # Final verification with remaining audio
    if buffer:
        result = await client.verify_speaker(
            voice_id=voice_id,
            audio=bytes(buffer),
        )
        if on_verification:
            on_verification(result)
        yield result


# ============================================================================
# Dialogue Workflows
# ============================================================================


async def synthesize_dialogue(
    client,
    lines: List[DialogueLine],
    character_voices: Optional[Dict[str, str]] = None,
    default_voice: str = "axiom_default",
    transition_pause_ms: int = 300,
) -> DialogueResult:
    """
    Synthesize a multi-character dialogue.

    Args:
        client: VoxClient instance
        lines: List of dialogue lines
        character_voices: Mapping of character name to voice ID
        default_voice: Default voice for unknown characters
        transition_pause_ms: Pause between different speakers

    Returns:
        DialogueResult
    """
    character_voices = character_voices or {}
    segments = []
    total_audio = bytearray()
    total_duration = 0.0
    voices_used = set()
    last_voice = None

    for line in lines:
        # Determine voice
        voice_id = line.voice_id
        if not voice_id and line.character:
            voice_id = character_voices.get(line.character, default_voice)
        voice_id = voice_id or default_voice

        voices_used.add(voice_id)

        # Add transition pause if voice changed
        if last_voice and last_voice != voice_id:
            pause_bytes = _generate_silence(transition_pause_ms, 24000)
            total_audio.extend(pause_bytes)
            total_duration += transition_pause_ms / 1000

        # Synthesize line
        result = await client.synthesize(
            text=line.text,
            voice_id=voice_id,
            emotion_preset=line.emotion,
        )

        segment_start = total_duration
        total_audio.extend(result.audio)
        total_duration += result.duration_seconds

        segments.append({
            "text": line.text,
            "character": line.character,
            "voice_id": voice_id,
            "start_seconds": segment_start,
            "duration_seconds": result.duration_seconds,
        })

        # Add pause after line
        if line.pause_after_ms > 0:
            pause_bytes = _generate_silence(line.pause_after_ms, 24000)
            total_audio.extend(pause_bytes)
            total_duration += line.pause_after_ms / 1000

        last_voice = voice_id

    return DialogueResult(
        audio=bytes(total_audio),
        lines=lines,
        voices_used=list(voices_used),
        total_duration_seconds=total_duration,
        segments=segments,
    )


def _generate_silence(duration_ms: int, sample_rate: int) -> bytes:
    """Generate silence audio."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return bytes(num_samples * 2)  # 16-bit silence


# ============================================================================
# Pipeline Builder
# ============================================================================


class WorkflowBuilder:
    """
    Builder for custom workflows.

    Example:
        workflow = (
            WorkflowBuilder("my_workflow")
            .add_step("check_consent", "check_consent", voice_id=voice_id)
            .add_step("synthesize", "synthesize", text=text)
            .add_step("quality_check", "check_quality", min_score=0.7)
            .on_failure("synthesize", "fallback_voice")
            .build()
        )
        result = await workflow.execute(client)
    """

    def __init__(self, name: str):
        """
        Initialize workflow builder.

        Args:
            name: Workflow name
        """
        self.name = name
        self._steps: List[WorkflowStep] = []
        self._error_handlers: Dict[str, str] = {}

    def add_step(
        self,
        name: str,
        operation: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        **params,
    ) -> "WorkflowBuilder":
        """
        Add a step to the workflow.

        Args:
            name: Step name
            operation: Operation to perform
            condition: Optional condition function
            **params: Operation parameters

        Returns:
            Self for chaining
        """
        step = WorkflowStep(
            name=name,
            operation=operation,
            params=params,
            condition=condition,
        )
        self._steps.append(step)
        return self

    def on_failure(self, step_name: str, handler_step: str) -> "WorkflowBuilder":
        """
        Set error handler for a step.

        Args:
            step_name: Step that might fail
            handler_step: Step to run on failure

        Returns:
            Self for chaining
        """
        self._error_handlers[step_name] = handler_step
        return self

    def build(self) -> "Workflow":
        """Build the workflow."""
        return Workflow(
            name=self.name,
            steps=self._steps,
            error_handlers=self._error_handlers,
        )


class Workflow:
    """Executable workflow."""

    def __init__(
        self,
        name: str,
        steps: List[WorkflowStep],
        error_handlers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize workflow.

        Args:
            name: Workflow name
            steps: List of workflow steps
            error_handlers: Mapping of step names to error handler step names
        """
        self.name = name
        self.steps = steps
        self.error_handlers = error_handlers or {}
        self._step_map = {step.name: step for step in steps}

    async def execute(
        self,
        client,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Execute the workflow.

        Args:
            client: VoxClient instance
            context: Initial context

        Returns:
            WorkflowResult
        """
        import time as time_module

        start_time = time_module.time()
        context = context or {}
        result = WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.RUNNING,
        )

        current_step_index = 0

        try:
            while current_step_index < len(self.steps):
                step = self.steps[current_step_index]

                # Check condition
                if step.condition and not step.condition(context):
                    logger.debug(f"Skipping step {step.name} (condition not met)")
                    current_step_index += 1
                    continue

                # Execute step
                try:
                    step_result = await self._execute_step(client, step, context)
                    result.step_results[step.name] = step_result
                    result.steps_completed.append(step.name)
                    context[f"{step.name}_result"] = step_result

                    if step.on_success and step.on_success in self._step_map:
                        current_step_index = self._find_step_index(step.on_success)
                    else:
                        current_step_index += 1

                except Exception as e:
                    logger.error(f"Step {step.name} failed: {e}")

                    if step.name in self.error_handlers:
                        handler_name = self.error_handlers[step.name]
                        current_step_index = self._find_step_index(handler_name)
                        context["error"] = str(e)
                    else:
                        raise

            result.status = WorkflowStatus.COMPLETED
            result.final_output = context.get(f"{self.steps[-1].name}_result")

        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.error = str(e)

        result.duration_ms = (time_module.time() - start_time) * 1000
        return result

    async def _execute_step(
        self,
        client,
        step: WorkflowStep,
        context: Dict[str, Any],
    ) -> Any:
        """Execute a single step."""
        # Resolve parameters from context
        params = {}
        for key, value in step.params.items():
            if isinstance(value, str) and value.startswith("$"):
                context_key = value[1:]
                params[key] = context.get(context_key)
            else:
                params[key] = value

        # Call client method
        method = getattr(client, step.operation, None)
        if method is None:
            raise ValueError(f"Unknown operation: {step.operation}")

        return await method(**params)

    def _find_step_index(self, step_name: str) -> int:
        """Find index of step by name."""
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                return i
        raise ValueError(f"Step not found: {step_name}")
