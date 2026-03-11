"""
PRIME Voice Streamer
--------------------

Real-time streaming voice output for PRIME.

Provides async streaming synthesis so PRIME can speak in real-time,
with sentence-level chunking, mode transitions, and live monitoring.

Architecture:
    Text → Sentence Segmentation → Per-Sentence Rendering → Audio Chunks → Output

The streamer supports:
- Progressive audio delivery (first chunk < 500ms)
- Sentence-boundary synchronization
- Mid-stream mode transitions
- Live quality monitoring
- Pause/resume/cancel controls
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from .identity import PrimeVoiceIdentityManager, get_identity_manager
from .models import (
    PrimeUtterance,
    PrimeVoiceState,
    SpeakingModeType,
    UtteranceType,
)
from .speaking_modes import SpeakingModeManager

logger = logging.getLogger(__name__)


# =============================================================================
# Stream Models
# =============================================================================

class StreamEventType(Enum):
    """Types of events in a PRIME voice stream."""
    STARTED = "started"
    CHUNK = "chunk"
    SENTENCE_BOUNDARY = "sentence_boundary"
    MODE_SWITCH = "mode_switch"
    PROGRESS = "progress"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"
    RESUMED = "resumed"
    CANCELLED = "cancelled"


@dataclass
class PrimeAudioChunk:
    """A single audio chunk from PRIME's voice stream."""
    data: Optional[bytes] = None         # Audio bytes (float32 WAV)
    chunk_index: int = 0
    timestamp_ms: float = 0.0
    duration_ms: float = 0.0

    # Sentence tracking
    sentence_index: int = 0
    sentence_text: str = ""
    is_sentence_start: bool = False
    is_sentence_end: bool = False

    # Voice state
    speaking_mode: SpeakingModeType = SpeakingModeType.CONVERSATIONAL
    sample_rate: int = 24000


@dataclass
class PrimeStreamEvent:
    """An event in PRIME's voice stream."""
    event_type: StreamEventType
    timestamp: float = field(default_factory=time.time)

    # Audio (for CHUNK events)
    chunk: Optional[PrimeAudioChunk] = None

    # Progress info
    sentences_completed: int = 0
    sentences_total: int = 0
    elapsed_ms: float = 0.0

    # Mode switch info
    from_mode: Optional[SpeakingModeType] = None
    to_mode: Optional[SpeakingModeType] = None

    # Error info
    error: Optional[str] = None

    # Metadata
    session_id: str = ""
    stream_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "stream_id": self.stream_id,
        }

        if self.chunk:
            result["chunk"] = {
                "chunk_index": self.chunk.chunk_index,
                "duration_ms": self.chunk.duration_ms,
                "sentence_index": self.chunk.sentence_index,
                "is_sentence_end": self.chunk.is_sentence_end,
                "speaking_mode": self.chunk.speaking_mode.value,
            }

        if self.event_type == StreamEventType.PROGRESS:
            result["progress"] = {
                "sentences_completed": self.sentences_completed,
                "sentences_total": self.sentences_total,
                "elapsed_ms": self.elapsed_ms,
            }

        if self.error:
            result["error"] = self.error

        if self.from_mode:
            result["mode_switch"] = {
                "from": self.from_mode.value,
                "to": self.to_mode.value if self.to_mode else None,
            }

        return result


@dataclass
class StreamConfig:
    """Configuration for PRIME voice streaming."""
    # Chunking
    chunk_size_bytes: int = 4096
    sample_rate: int = 24000
    max_first_chunk_ms: int = 500

    # Buffering
    min_buffer_sentences: int = 1
    max_buffer_sentences: int = 3

    # Mode detection
    auto_detect_mode_per_sentence: bool = True

    # Controls
    allow_pause: bool = True
    allow_cancel: bool = True

    # Monitoring
    report_progress: bool = True
    progress_interval: int = 1  # Report every N sentences


# =============================================================================
# Sentence Segmenter
# =============================================================================

def segment_sentences(text: str) -> List[str]:
    """
    Split text into sentences for progressive streaming.

    Uses simple heuristics - splits on sentence-ending punctuation
    while handling common abbreviations and edge cases.
    """
    import re

    # Split on sentence-ending punctuation followed by space
    raw = re.split(r'(?<=[.!?])\s+', text.strip())

    # Filter empty strings and merge very short fragments
    sentences = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        # Merge very short fragments with previous sentence
        if sentences and len(s.split()) <= 2 and not s.endswith(('.', '!', '?')):
            sentences[-1] += " " + s
        else:
            sentences.append(s)

    return sentences if sentences else [text.strip()]


# =============================================================================
# PRIME Voice Streamer
# =============================================================================

class PrimeVoiceStreamer:
    """
    Real-time streaming voice output for PRIME.

    Provides async iteration over audio chunks as PRIME speaks,
    with sentence-level control and mode transitions.

    Usage:
        streamer = PrimeVoiceStreamer()

        # Stream PRIME speaking
        async for event in streamer.stream("Systems nominal. Ready for deployment."):
            if event.event_type == StreamEventType.CHUNK:
                play_audio(event.chunk.data)
            elif event.event_type == StreamEventType.SENTENCE_BOUNDARY:
                print(f"Finished: {event.chunk.sentence_text}")

        # Stream with mode override
        async for event in streamer.stream(
            "Critical alert: database connection lost!",
            mode=SpeakingModeType.ALERT,
        ):
            handle_event(event)
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        identity_manager: Optional[PrimeVoiceIdentityManager] = None,
    ):
        self.config = config or StreamConfig()
        self._identity = identity_manager or get_identity_manager()
        self._mode_manager = SpeakingModeManager()
        self._state = PrimeVoiceState.READY
        self._active_stream_id: Optional[str] = None
        self._cancel_requested = False
        self._pause_requested = False
        self._synthesizer = None

    def _get_synthesizer(self):
        """Lazy-load synthesizer."""
        if self._synthesizer is None:
            try:
                from axiom_vox.synthesis import VoxSynthesizer
                self._synthesizer = VoxSynthesizer(model_size="small")
            except ImportError:
                self._synthesizer = _PlaceholderStreamSynthesizer()
        return self._synthesizer

    async def stream(
        self,
        text: str,
        mode: Optional[SpeakingModeType] = None,
        context: Optional[Dict[str, Any]] = None,
        on_event: Optional[Callable] = None,
    ) -> AsyncIterator[PrimeStreamEvent]:
        """
        Stream PRIME speaking the given text.

        Yields PrimeStreamEvent objects as audio is generated.

        Args:
            text: Text for PRIME to speak
            mode: Explicit speaking mode (auto-detected per sentence if None)
            context: Optional context for mode detection
            on_event: Optional callback for each event

        Yields:
            PrimeStreamEvent objects (STARTED, CHUNK, SENTENCE_BOUNDARY, COMPLETED)
        """
        stream_id = f"prime_stream_{uuid.uuid4().hex[:8]}"
        session_id = f"prime_session_{uuid.uuid4().hex[:8]}"
        self._active_stream_id = stream_id
        self._cancel_requested = False
        self._pause_requested = False
        self._state = PrimeVoiceState.STREAMING

        start_time = time.time()

        # Segment text into sentences
        sentences = segment_sentences(text)

        # Emit STARTED event
        started_event = PrimeStreamEvent(
            event_type=StreamEventType.STARTED,
            sentences_total=len(sentences),
            session_id=session_id,
            stream_id=stream_id,
        )
        if on_event:
            on_event(started_event)
        yield started_event

        chunk_index = 0

        try:
            for sent_idx, sentence in enumerate(sentences):
                # Check for cancellation
                if self._cancel_requested:
                    cancel_event = PrimeStreamEvent(
                        event_type=StreamEventType.CANCELLED,
                        sentences_completed=sent_idx,
                        sentences_total=len(sentences),
                        elapsed_ms=(time.time() - start_time) * 1000,
                        session_id=session_id,
                        stream_id=stream_id,
                    )
                    if on_event:
                        on_event(cancel_event)
                    yield cancel_event
                    return

                # Handle pause
                while self._pause_requested:
                    if self._state != PrimeVoiceState.PAUSED:
                        self._state = PrimeVoiceState.PAUSED
                        pause_event = PrimeStreamEvent(
                            event_type=StreamEventType.PAUSED,
                            session_id=session_id,
                            stream_id=stream_id,
                        )
                        if on_event:
                            on_event(pause_event)
                        yield pause_event
                    await asyncio.sleep(0.1)

                if self._state == PrimeVoiceState.PAUSED:
                    self._state = PrimeVoiceState.STREAMING
                    resume_event = PrimeStreamEvent(
                        event_type=StreamEventType.RESUMED,
                        session_id=session_id,
                        stream_id=stream_id,
                    )
                    if on_event:
                        on_event(resume_event)
                    yield resume_event

                # Detect mode for this sentence (if auto-detect enabled)
                current_mode = mode
                if current_mode is None and self.config.auto_detect_mode_per_sentence:
                    detection = self._mode_manager.detect_mode(sentence, context)
                    if detection.confidence >= 0.6:
                        if detection.detected_mode != self._mode_manager.current_mode:
                            old_mode = self._mode_manager.current_mode
                            self._mode_manager.switch_mode(detection.detected_mode)

                            mode_event = PrimeStreamEvent(
                                event_type=StreamEventType.MODE_SWITCH,
                                from_mode=old_mode,
                                to_mode=detection.detected_mode,
                                session_id=session_id,
                                stream_id=stream_id,
                            )
                            if on_event:
                                on_event(mode_event)
                            yield mode_event

                        current_mode = detection.detected_mode
                    else:
                        current_mode = self._mode_manager.current_mode
                elif current_mode is None:
                    current_mode = self._mode_manager.current_mode

                # Synthesize sentence
                audio_data, duration = await self._synthesize_async(
                    sentence, current_mode
                )

                # Emit audio chunk(s)
                if audio_data:
                    # Split into chunks if needed
                    chunks = self._split_audio(
                        audio_data, self.config.chunk_size_bytes
                    )
                    for i, chunk_data in enumerate(chunks):
                        chunk = PrimeAudioChunk(
                            data=chunk_data,
                            chunk_index=chunk_index,
                            timestamp_ms=(time.time() - start_time) * 1000,
                            duration_ms=(duration / len(chunks)) * 1000,
                            sentence_index=sent_idx,
                            sentence_text=sentence,
                            is_sentence_start=(i == 0),
                            is_sentence_end=(i == len(chunks) - 1),
                            speaking_mode=current_mode,
                            sample_rate=self.config.sample_rate,
                        )

                        chunk_event = PrimeStreamEvent(
                            event_type=StreamEventType.CHUNK,
                            chunk=chunk,
                            session_id=session_id,
                            stream_id=stream_id,
                        )
                        if on_event:
                            on_event(chunk_event)
                        yield chunk_event
                        chunk_index += 1
                else:
                    # No audio data - still emit a placeholder chunk
                    chunk = PrimeAudioChunk(
                        data=None,
                        chunk_index=chunk_index,
                        timestamp_ms=(time.time() - start_time) * 1000,
                        duration_ms=duration * 1000,
                        sentence_index=sent_idx,
                        sentence_text=sentence,
                        is_sentence_start=True,
                        is_sentence_end=True,
                        speaking_mode=current_mode,
                    )

                    chunk_event = PrimeStreamEvent(
                        event_type=StreamEventType.CHUNK,
                        chunk=chunk,
                        session_id=session_id,
                        stream_id=stream_id,
                    )
                    if on_event:
                        on_event(chunk_event)
                    yield chunk_event
                    chunk_index += 1

                # Emit sentence boundary
                boundary_event = PrimeStreamEvent(
                    event_type=StreamEventType.SENTENCE_BOUNDARY,
                    chunk=PrimeAudioChunk(
                        sentence_index=sent_idx,
                        sentence_text=sentence,
                        is_sentence_end=True,
                        speaking_mode=current_mode,
                    ),
                    sentences_completed=sent_idx + 1,
                    sentences_total=len(sentences),
                    session_id=session_id,
                    stream_id=stream_id,
                )
                if on_event:
                    on_event(boundary_event)
                yield boundary_event

                # Progress reporting
                if (self.config.report_progress and
                        (sent_idx + 1) % self.config.progress_interval == 0):
                    progress_event = PrimeStreamEvent(
                        event_type=StreamEventType.PROGRESS,
                        sentences_completed=sent_idx + 1,
                        sentences_total=len(sentences),
                        elapsed_ms=(time.time() - start_time) * 1000,
                        session_id=session_id,
                        stream_id=stream_id,
                    )
                    if on_event:
                        on_event(progress_event)
                    yield progress_event

            # Emit COMPLETED
            completed_event = PrimeStreamEvent(
                event_type=StreamEventType.COMPLETED,
                sentences_completed=len(sentences),
                sentences_total=len(sentences),
                elapsed_ms=(time.time() - start_time) * 1000,
                session_id=session_id,
                stream_id=stream_id,
            )
            if on_event:
                on_event(completed_event)
            yield completed_event

        except Exception as e:
            error_event = PrimeStreamEvent(
                event_type=StreamEventType.ERROR,
                error=str(e),
                elapsed_ms=(time.time() - start_time) * 1000,
                session_id=session_id,
                stream_id=stream_id,
            )
            if on_event:
                on_event(error_event)
            yield error_event

        finally:
            self._state = PrimeVoiceState.READY
            self._active_stream_id = None

    # -------------------------------------------------------------------------
    # Controls
    # -------------------------------------------------------------------------

    def pause(self) -> bool:
        """Pause the active stream."""
        if self._state == PrimeVoiceState.STREAMING:
            self._pause_requested = True
            return True
        return False

    def resume(self) -> bool:
        """Resume a paused stream."""
        if self._pause_requested:
            self._pause_requested = False
            return True
        return False

    def cancel(self) -> bool:
        """Cancel the active stream."""
        if self._active_stream_id:
            self._cancel_requested = True
            return True
        return False

    @property
    def state(self) -> PrimeVoiceState:
        return self._state

    @property
    def is_streaming(self) -> bool:
        return self._state == PrimeVoiceState.STREAMING

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    async def _synthesize_async(
        self,
        text: str,
        mode: SpeakingModeType,
    ) -> tuple:
        """Synthesize a sentence asynchronously."""
        synthesizer = self._get_synthesizer()

        # Get PRIME's voice parameters with mode adjustments
        identity = self._identity.get_identity()
        mode_profile = self._mode_manager.get_mode_profile(mode)

        voice_params = {
            "voice_id": identity.vox_voice_id,
            "speaking_rate": identity.vocal_dna.target_speaking_rate * mode_profile.rate_multiplier,
            "pitch": mode_profile.pitch_shift,
            "emotion": identity.emotion_preset,
        }

        try:
            # Check for async synthesize method
            if hasattr(synthesizer, 'synthesize_async'):
                result = await synthesizer.synthesize_async(text=text, **voice_params)
            else:
                # Run sync synthesis in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: synthesizer.synthesize(text=text, **voice_params),
                )

            if hasattr(result, 'audio_data'):
                return result.audio_data, result.duration_seconds or self._estimate_duration(text)
            elif isinstance(result, dict):
                return result.get("audio_data"), result.get("duration_seconds", self._estimate_duration(text))
            else:
                return None, self._estimate_duration(text)

        except Exception as e:
            logger.warning(f"Stream synthesis failed for sentence: {e}")
            return None, self._estimate_duration(text)

    def _split_audio(self, audio_data: bytes, chunk_size: int) -> List[bytes]:
        """Split audio data into chunks."""
        if len(audio_data) <= chunk_size:
            return [audio_data]

        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunks.append(audio_data[i:i + chunk_size])
        return chunks

    def _estimate_duration(self, text: str) -> float:
        """Estimate duration from text."""
        words = len(text.split())
        return max(0.3, words / 150 * 60)


# =============================================================================
# Placeholder Stream Synthesizer
# =============================================================================

class _PlaceholderStreamSynthesizer:
    """Placeholder when real synthesizer is not available."""

    def synthesize(self, text: str, **kwargs):
        words = len(text.split())
        duration = max(0.3, words / 150 * 60)
        return type("Result", (), {
            "audio_data": None,
            "duration_seconds": duration,
        })()


# =============================================================================
# Convenience Functions
# =============================================================================

_default_streamer: Optional[PrimeVoiceStreamer] = None


def get_streamer(
    config: Optional[StreamConfig] = None,
) -> PrimeVoiceStreamer:
    """Get or create the default PRIME voice streamer."""
    global _default_streamer
    if _default_streamer is None:
        _default_streamer = PrimeVoiceStreamer(config)
    return _default_streamer


async def prime_stream(
    text: str,
    mode: Optional[SpeakingModeType] = None,
    context: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[PrimeStreamEvent]:
    """
    Stream PRIME speaking the given text.

    Usage:
        from axiom_vox.prime_voice import prime_stream

        async for event in prime_stream("All systems operational."):
            if event.event_type == StreamEventType.CHUNK:
                play_audio(event.chunk.data)
    """
    streamer = get_streamer()
    async for event in streamer.stream(text, mode=mode, context=context):
        yield event
