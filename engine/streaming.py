"""
VØX Streaming Infrastructure
-----------------------------

True streaming TTS with progressive audio generation.

Key Components:
    - StreamConfig: Configuration for streaming behavior
    - AudioChunk: Individual audio chunk with metadata
    - StreamMessage: Protocol messages for WebSocket
    - StreamSession: Session state tracking
    - SentenceSegmenter: Text segmentation for progressive synthesis
    - StreamingSynthesizer: True progressive audio generation
    - StreamManager: Orchestrates streaming sessions

Usage:
    from axiom_vox.streaming import StreamManager, get_stream_manager

    manager = get_stream_manager()
    session = manager.create_session(text="Hello world", voice_id="default")

    # Run governance check
    await manager.run_governance(session, context)

    # Stream audio
    async for message in manager.stream(session):
        if message.type == MessageType.CHUNK:
            play_audio(message.chunk.data)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
)

if TYPE_CHECKING:
    from axiom_vox.synthesis import VoxSynthesizer, VoiceConfig
    from axiom_vox.vox_governor import VoxGovernor
    from axiom_vox.prosody_guardrails import EmotionalIntent
    from axiom_vox.analytics.streaming_collector import StreamingAnalyticsCollector

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class StreamConfig:
    """Configuration for streaming TTS."""

    # Audio parameters
    chunk_size_bytes: int = 4096  # Standard chunk size (~170ms at 24kHz 16-bit)
    sample_rate: int = 24000  # Qwen3-TTS default

    # Latency targets
    max_first_chunk_ms: int = 500  # Target: first audio within 500ms
    sentence_lookahead: int = 1  # Sentences to buffer ahead

    # Buffering behavior
    min_buffer_chunks: int = 2  # Minimum chunks before streaming
    max_buffer_chunks: int = 10  # Maximum buffer size

    # Error handling
    retry_on_error: bool = True
    max_retries: int = 3
    retry_delay_ms: int = 100

    # Progress reporting
    report_progress: bool = True
    progress_interval_chunks: int = 5  # Report every N chunks


# ============================================================================
# MESSAGE TYPES
# ============================================================================

class MessageType(str, Enum):
    """WebSocket message types."""

    # Lifecycle messages
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"

    # Audio messages
    CHUNK = "chunk"
    SENTENCE_BOUNDARY = "sentence_boundary"

    # Governance messages
    GOVERNANCE_PASSED = "governance_passed"
    GOVERNANCE_REPAIRED = "governance_repaired"
    GOVERNANCE_REFUSED = "governance_refused"

    # Multi-voice messages (v0.9.0)
    VOICE_SWITCH = "voice_switch"

    # Client control (WebSocket only)
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    PAUSED = "paused"
    RESUMED = "resumed"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AudioChunk:
    """Individual audio chunk with metadata."""

    data: bytes
    index: int
    timestamp_ms: float
    duration_ms: float
    is_sentence_end: bool = False
    sentence_index: Optional[int] = None

    # Multi-voice support (v0.9.0)
    voice_id: Optional[str] = None
    is_voice_transition: bool = False

    # For reconstruction
    sample_rate: int = 24000
    format: str = "wav"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict (excluding binary data)."""
        return {
            "index": self.index,
            "timestamp_ms": self.timestamp_ms,
            "duration_ms": self.duration_ms,
            "is_sentence_end": self.is_sentence_end,
            "sentence_index": self.sentence_index,
            "voice_id": self.voice_id,
            "is_voice_transition": self.is_voice_transition,
            "size_bytes": len(self.data),
            "sample_rate": self.sample_rate,
            "format": self.format,
        }


@dataclass
class StreamMessage:
    """Protocol message for WebSocket communication."""

    type: MessageType
    request_id: str
    timestamp: float = field(default_factory=time.time)

    # Payload varies by type
    chunk: Optional[AudioChunk] = None
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    governance_report: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    # Multi-voice support (v0.9.0)
    voice_switch: Optional[Dict[str, Any]] = None  # {from_voice, to_voice, ...}

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "type": self.type.value,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }

        if self.chunk:
            result["chunk"] = self.chunk.to_dict()
            # Note: Audio data is sent separately as binary
        if self.progress:
            result["progress"] = self.progress
        if self.error:
            result["error"] = self.error
        if self.governance_report:
            result["governance_report"] = self.governance_report
        if self.metadata:
            result["metadata"] = self.metadata
        if self.voice_switch:
            result["voice_switch"] = self.voice_switch

        return result

    def to_json_string(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_json())


# ============================================================================
# STREAM SESSION
# ============================================================================

@dataclass
class StreamSession:
    """
    Tracks state for a single streaming session.

    Lifecycle:
        PENDING -> GOVERNING -> LOADING_VOICE -> SYNTHESIZING -> STREAMING -> COMPLETED
                                                                          -> ERROR
                                                                          -> CANCELLED
    """

    class State(str, Enum):
        PENDING = "pending"
        GOVERNING = "governing"
        LOADING_VOICE = "loading_voice"
        SYNTHESIZING = "synthesizing"
        STREAMING = "streaming"
        COMPLETED = "completed"
        ERROR = "error"
        CANCELLED = "cancelled"
        PAUSED = "paused"

    request_id: str
    text: str
    voice_id: str
    state: State = State.PENDING

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    first_chunk_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Progress tracking
    total_sentences: int = 0
    sentences_completed: int = 0
    chunks_sent: int = 0
    bytes_sent: int = 0

    # Governance
    governed_text: Optional[str] = None
    governance_report: Optional[Dict[str, Any]] = None

    # Error tracking
    error_message: Optional[str] = None
    retries: int = 0

    def get_latency_to_first_chunk_ms(self) -> Optional[float]:
        """Calculate time from start to first chunk."""
        if self.started_at and self.first_chunk_at:
            return (self.first_chunk_at - self.started_at) * 1000
        return None

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress report."""
        return {
            "state": self.state.value,
            "sentences_completed": self.sentences_completed,
            "total_sentences": self.total_sentences,
            "chunks_sent": self.chunks_sent,
            "bytes_sent": self.bytes_sent,
            "latency_to_first_chunk_ms": self.get_latency_to_first_chunk_ms(),
            "elapsed_ms": (time.time() - self.started_at) * 1000 if self.started_at else 0,
        }


# ============================================================================
# SENTENCE SEGMENTATION
# ============================================================================

class SentenceSegmenter:
    """
    Segments text into sentences for progressive synthesis.

    Enables streaming by allowing synthesis to start on first sentence
    while subsequent sentences are being processed.
    """

    # Sentence-ending punctuation
    SENTENCE_ENDINGS = {'.', '!', '?'}

    # Abbreviations that shouldn't end sentences
    ABBREVIATIONS = {
        'mr.', 'mrs.', 'ms.', 'dr.', 'prof.',
        'inc.', 'ltd.', 'corp.',
        'vs.', 'etc.', 'e.g.', 'i.e.',
        'st.', 'ave.', 'blvd.',
        'no.', 'vol.', 'rev.',
        'jan.', 'feb.', 'mar.', 'apr.', 'jun.',
        'jul.', 'aug.', 'sep.', 'oct.', 'nov.', 'dec.',
    }

    @classmethod
    def segment(cls, text: str) -> List[str]:
        """
        Segment text into sentences.

        Returns list of sentences preserving original spacing.
        """
        if not text or not text.strip():
            return []

        sentences = []
        current = []

        words = text.split()
        for i, word in enumerate(words):
            current.append(word)

            # Check if this word ends a sentence
            if cls._ends_sentence(word, words, i):
                sentence = ' '.join(current)
                sentences.append(sentence)
                current = []

        # Don't lose remaining words
        if current:
            sentences.append(' '.join(current))

        return sentences

    @classmethod
    def _ends_sentence(cls, word: str, words: List[str], index: int) -> bool:
        """Check if word ends a sentence."""
        if not word:
            return False

        # Must end with sentence punctuation
        if word[-1] not in cls.SENTENCE_ENDINGS:
            return False

        # Check for abbreviations
        word_lower = word.lower()
        if word_lower in cls.ABBREVIATIONS:
            return False

        # Check for ellipsis (...)
        if word.endswith('...'):
            return True

        # Check next word capitalization (if exists)
        if index + 1 < len(words):
            next_word = words[index + 1]
            if next_word and next_word[0].isupper():
                return True
            # If next word starts lowercase, might be continuation
            return False

        # End of text
        if index == len(words) - 1:
            return True

        return False

    @classmethod
    def count_sentences(cls, text: str) -> int:
        """Count number of sentences without full segmentation."""
        return len(cls.segment(text))


# ============================================================================
# STREAMING SYNTHESIZER
# ============================================================================

class StreamingSynthesizer:
    """
    True streaming synthesizer with progressive audio generation.

    Key innovation: Instead of generating all audio then chunking,
    this synthesizes sentence-by-sentence and streams chunks as
    they become available.

    Architecture:
        1. Segment text into sentences
        2. Start synthesizing first sentence immediately
        3. Buffer minimum chunks before streaming
        4. Stream chunks while synthesizing subsequent sentences
        5. Signal sentence boundaries for client synchronization
    """

    def __init__(
        self,
        synthesizer: "VoxSynthesizer",
        config: Optional[StreamConfig] = None,
    ):
        self.synthesizer = synthesizer
        self.config = config or StreamConfig()
        self.segmenter = SentenceSegmenter()

    async def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        voice_config: Optional["VoiceConfig"] = None,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> AsyncIterator[AudioChunk]:
        """
        True streaming synthesis with progressive generation.

        Yields AudioChunks as they become available.
        First chunk target: < 500ms latency.
        """
        # Segment into sentences
        sentences = self.segmenter.segment(text)
        if not sentences:
            return

        total_sentences = len(sentences)
        chunk_index = 0
        start_time = time.time()
        first_chunk_time = None

        # Process sentences
        for sentence_idx, sentence in enumerate(sentences):
            try:
                # Generate audio for this sentence
                result = self.synthesizer.synthesize(
                    text=sentence,
                    voice=voice_config,
                )

                if not result.success or not result.audio_data:
                    logger.warning(f"Sentence {sentence_idx} synthesis failed: {result.error}")
                    continue

                # Chunk this sentence's audio
                audio_data = result.audio_data

                for offset in range(0, len(audio_data), self.config.chunk_size_bytes):
                    chunk_data = audio_data[offset:offset + self.config.chunk_size_bytes]

                    # Calculate chunk duration based on bytes
                    # For 24kHz 32-bit float: 4 bytes per sample
                    bytes_per_sample = 4
                    samples_in_chunk = len(chunk_data) / bytes_per_sample
                    chunk_duration_ms = (samples_in_chunk / self.config.sample_rate) * 1000

                    is_last_chunk = offset + self.config.chunk_size_bytes >= len(audio_data)

                    chunk = AudioChunk(
                        data=chunk_data,
                        index=chunk_index,
                        timestamp_ms=(time.time() - start_time) * 1000,
                        duration_ms=chunk_duration_ms,
                        is_sentence_end=is_last_chunk,
                        sentence_index=sentence_idx if is_last_chunk else None,
                        sample_rate=self.config.sample_rate,
                    )

                    # Track first chunk latency
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        latency = (first_chunk_time - start_time) * 1000
                        logger.info(f"First chunk latency: {latency:.1f}ms")

                    yield chunk
                    chunk_index += 1

                    # Small delay to enable interleaving
                    await asyncio.sleep(0.001)

                # Report progress
                if on_progress and (sentence_idx + 1) % 2 == 0:
                    on_progress({
                        "sentences_completed": sentence_idx + 1,
                        "total_sentences": total_sentences,
                        "chunks_sent": chunk_index,
                    })

            except Exception as e:
                logger.error(f"Error synthesizing sentence {sentence_idx}: {e}")
                # Continue with next sentence on error
                continue

    async def synthesize_stream_placeholder(
        self,
        text: str,
        voice_id: str,
    ) -> AsyncIterator[AudioChunk]:
        """
        Placeholder streaming when model not loaded.

        Generates silence with timing beeps for testing.
        """
        try:
            import numpy as np
            HAS_NUMPY = True
        except ImportError:
            HAS_NUMPY = False
            logger.warning("numpy not available for placeholder synthesis")
            return

        sentences = self.segmenter.segment(text)
        chunk_index = 0
        start_time = time.time()

        for sentence_idx, sentence in enumerate(sentences):
            # Estimate duration: ~100ms per word
            word_count = len(sentence.split())
            duration_seconds = max(0.5, word_count * 0.1)

            # Generate placeholder audio
            samples = int(self.config.sample_rate * duration_seconds)
            audio = np.zeros(samples, dtype=np.float32)

            # Add small beep at start of each sentence (different pitch per sentence)
            beep_samples = int(self.config.sample_rate * 0.05)
            t = np.linspace(0, 0.05, beep_samples)
            frequency = 440 * (sentence_idx + 1)  # A4, A5, A6, etc.
            audio[:beep_samples] = 0.2 * np.sin(2 * np.pi * frequency * t)

            # Convert to bytes
            audio_bytes = audio.tobytes()

            # Chunk it
            for offset in range(0, len(audio_bytes), self.config.chunk_size_bytes):
                chunk_data = audio_bytes[offset:offset + self.config.chunk_size_bytes]
                is_last = offset + self.config.chunk_size_bytes >= len(audio_bytes)

                # Calculate duration
                bytes_per_sample = 4  # float32
                samples_in_chunk = len(chunk_data) / bytes_per_sample
                chunk_duration_ms = (samples_in_chunk / self.config.sample_rate) * 1000

                chunk = AudioChunk(
                    data=chunk_data,
                    index=chunk_index,
                    timestamp_ms=(time.time() - start_time) * 1000,
                    duration_ms=chunk_duration_ms,
                    is_sentence_end=is_last,
                    sentence_index=sentence_idx if is_last else None,
                    sample_rate=self.config.sample_rate,
                )

                yield chunk
                chunk_index += 1

                await asyncio.sleep(0.01)  # Simulate synthesis time


# ============================================================================
# STREAM MANAGER
# ============================================================================

class StreamManager:
    """
    Manages streaming TTS sessions.

    Responsibilities:
        - Session lifecycle management
        - Governance integration (pre-flight check)
        - Voice/adapter loading
        - Progress tracking
        - Error recovery
    """

    def __init__(
        self,
        synthesizer: "VoxSynthesizer",
        governor: "VoxGovernor",
        config: Optional[StreamConfig] = None,
    ):
        self.synthesizer = synthesizer
        self.governor = governor
        self.config = config or StreamConfig()
        self.streaming_synth = StreamingSynthesizer(synthesizer, config)

        # Active sessions
        self._sessions: Dict[str, StreamSession] = {}

    def create_session(
        self,
        text: str,
        voice_id: str,
        request_id: Optional[str] = None,
    ) -> StreamSession:
        """Create a new streaming session."""
        request_id = request_id or f"stream_{uuid.uuid4().hex[:12]}"

        session = StreamSession(
            request_id=request_id,
            text=text,
            voice_id=voice_id,
        )

        self._sessions[request_id] = session
        return session

    def get_session(self, request_id: str) -> Optional[StreamSession]:
        """Get session by ID."""
        return self._sessions.get(request_id)

    def remove_session(self, request_id: str) -> None:
        """Remove a completed session."""
        self._sessions.pop(request_id, None)

    def cancel_session(self, request_id: str) -> bool:
        """Cancel a streaming session."""
        session = self._sessions.get(request_id)
        if session and session.state in (
            StreamSession.State.PENDING,
            StreamSession.State.GOVERNING,
            StreamSession.State.SYNTHESIZING,
            StreamSession.State.STREAMING,
            StreamSession.State.PAUSED,
        ):
            session.state = StreamSession.State.CANCELLED
            return True
        return False

    async def run_governance(
        self,
        session: StreamSession,
        context: Optional[Dict[str, Any]] = None,
        emotional_intent: Optional["EmotionalIntent"] = None,
    ) -> bool:
        """
        Run pre-flight governance check.

        Returns True if synthesis should proceed.
        Sets session.governed_text and session.governance_report.
        """
        session.state = StreamSession.State.GOVERNING

        try:
            result = self.governor.govern(
                text=session.text,
                voice_id=session.voice_id,
                context=context,
                emotional_intent=emotional_intent,
            )

            # Convert to dict for storage
            if hasattr(result, 'to_dict'):
                session.governance_report = result.to_dict()
            else:
                session.governance_report = {
                    "action": result.action.value if hasattr(result.action, 'value') else str(result.action),
                    "governed_text": result.governed_text,
                }

            if result.action.value == "refuse":
                session.state = StreamSession.State.ERROR
                refusal_reason = getattr(result, 'refusal_reason', 'Content refused by governance')
                session.error_message = f"Governance refused: {refusal_reason}"
                return False

            # Use governed (possibly repaired) text
            session.governed_text = result.governed_text
            session.total_sentences = SentenceSegmenter.count_sentences(session.governed_text)

            return True

        except Exception as e:
            logger.error(f"Governance check failed: {e}")
            session.state = StreamSession.State.ERROR
            session.error_message = f"Governance error: {str(e)}"
            return False

    async def load_voice(self, session: StreamSession) -> bool:
        """
        Load voice adapter if needed.

        For clone_* voices, loads the LoRA adapter.
        """
        session.state = StreamSession.State.LOADING_VOICE

        if session.voice_id.startswith("clone_"):
            try:
                success = self.synthesizer.load_adapter(session.voice_id)
                if not success:
                    session.state = StreamSession.State.ERROR
                    session.error_message = f"Failed to load voice adapter: {session.voice_id}"
                    return False
            except Exception as e:
                logger.error(f"Voice loading failed: {e}")
                session.state = StreamSession.State.ERROR
                session.error_message = f"Voice loading error: {str(e)}"
                return False

        return True

    async def stream(
        self,
        session: StreamSession,
        voice_config: Optional["VoiceConfig"] = None,
        enable_analytics: bool = True,
    ) -> AsyncIterator[StreamMessage]:
        """
        Main streaming entry point.

        Yields StreamMessages for the session lifecycle.

        Args:
            session: StreamSession to process
            voice_config: Optional voice configuration override
            enable_analytics: Enable streaming analytics (default True)
        """
        session.started_at = time.time()

        # Initialize streaming analytics
        streaming_collector: Optional["StreamingAnalyticsCollector"] = None
        if enable_analytics:
            try:
                from axiom_vox.analytics.streaming_collector import get_streaming_collector
                streaming_collector = get_streaming_collector()
                streaming_collector.start_session(session.request_id, session.voice_id)
            except ImportError:
                logger.debug("Streaming analytics not available")
            except Exception as e:
                logger.warning(f"Failed to start streaming analytics: {e}")

        # Send started message
        yield StreamMessage(
            type=MessageType.STARTED,
            request_id=session.request_id,
            metadata={
                "text_length": len(session.text),
                "voice_id": session.voice_id,
                "total_sentences": session.total_sentences,
                "sample_rate": self.config.sample_rate,
                "format": "wav",
            },
        )

        # Send governance report
        if session.governance_report:
            action = session.governance_report.get("action", "allow")
            msg_type = {
                "allow": MessageType.GOVERNANCE_PASSED,
                "repair": MessageType.GOVERNANCE_REPAIRED,
                "warn": MessageType.GOVERNANCE_PASSED,
            }.get(action, MessageType.GOVERNANCE_PASSED)

            yield StreamMessage(
                type=msg_type,
                request_id=session.request_id,
                governance_report=session.governance_report,
            )

        # Start streaming synthesis
        session.state = StreamSession.State.STREAMING

        try:
            text_to_synthesize = session.governed_text or session.text

            # Choose synthesizer based on model availability
            if self.synthesizer._loaded:
                stream_gen = self.streaming_synth.synthesize_stream(
                    text=text_to_synthesize,
                    voice_id=session.voice_id,
                    voice_config=voice_config,
                )
            else:
                # Fallback to placeholder
                logger.info("Using placeholder streaming synthesis")
                stream_gen = self.streaming_synth.synthesize_stream_placeholder(
                    text=text_to_synthesize,
                    voice_id=session.voice_id,
                )

            async for chunk in stream_gen:
                # Check for cancellation
                if session.state == StreamSession.State.CANCELLED:
                    yield StreamMessage(
                        type=MessageType.CANCELLED,
                        request_id=session.request_id,
                    )
                    return

                # Handle pause
                while session.state == StreamSession.State.PAUSED:
                    await asyncio.sleep(0.1)
                    if session.state == StreamSession.State.CANCELLED:
                        yield StreamMessage(
                            type=MessageType.CANCELLED,
                            request_id=session.request_id,
                        )
                        return

                # Track timing
                if session.first_chunk_at is None:
                    session.first_chunk_at = time.time()

                # Update counters
                session.chunks_sent += 1
                session.bytes_sent += len(chunk.data)

                if chunk.is_sentence_end:
                    session.sentences_completed += 1

                # Yield chunk message
                yield StreamMessage(
                    type=MessageType.CHUNK,
                    request_id=session.request_id,
                    chunk=chunk,
                )

                # Record chunk to streaming analytics
                if streaming_collector:
                    try:
                        streaming_collector.record_chunk(
                            session_id=session.request_id,
                            chunk_index=chunk.index,
                            chunk_data=chunk.data,
                            timestamp_ms=chunk.timestamp_ms,
                            duration_ms=chunk.duration_ms,
                            is_sentence_end=chunk.is_sentence_end,
                            sentence_index=chunk.sentence_index,
                            sample_rate=chunk.sample_rate,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to record chunk analytics: {e}")

                # Send sentence boundary for client sync
                if chunk.is_sentence_end:
                    yield StreamMessage(
                        type=MessageType.SENTENCE_BOUNDARY,
                        request_id=session.request_id,
                        metadata={
                            "sentence_index": chunk.sentence_index,
                            "sentences_remaining": session.total_sentences - session.sentences_completed,
                        },
                    )

                # Periodic progress updates
                if self.config.report_progress and session.chunks_sent % self.config.progress_interval_chunks == 0:
                    yield StreamMessage(
                        type=MessageType.PROGRESS,
                        request_id=session.request_id,
                        progress=session.get_progress(),
                    )

            # Completed successfully
            session.state = StreamSession.State.COMPLETED
            session.completed_at = time.time()

            # End streaming analytics
            streaming_metrics = None
            if streaming_collector:
                try:
                    streaming_metrics = streaming_collector.end_session(session.request_id)
                    if streaming_metrics:
                        logger.debug(
                            f"Streaming session {session.request_id}: "
                            f"chunks={streaming_metrics.total_chunks}, "
                            f"stutters={streaming_metrics.stutters_detected}, "
                            f"rtf={streaming_metrics.real_time_factor:.3f}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to end streaming analytics: {e}")

            yield StreamMessage(
                type=MessageType.COMPLETED,
                request_id=session.request_id,
                metadata={
                    "total_chunks": session.chunks_sent,
                    "total_bytes": session.bytes_sent,
                    "total_sentences": session.sentences_completed,
                    "duration_ms": (session.completed_at - session.started_at) * 1000,
                    "latency_to_first_chunk_ms": session.get_latency_to_first_chunk_ms(),
                    "streaming_analytics": streaming_metrics.to_dict() if streaming_metrics else None,
                },
            )

        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            session.state = StreamSession.State.ERROR
            session.error_message = str(e)

            # End streaming analytics on error
            if streaming_collector:
                try:
                    streaming_collector.end_session(session.request_id)
                except Exception:
                    pass

            yield StreamMessage(
                type=MessageType.ERROR,
                request_id=session.request_id,
                error=str(e),
            )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    """Get or create the global stream manager."""
    global _stream_manager
    if _stream_manager is None:
        from axiom_vox.synthesis import get_synthesizer
        from axiom_vox.vox_governor import get_governor

        _stream_manager = StreamManager(
            synthesizer=get_synthesizer(),
            governor=get_governor(),
        )
    return _stream_manager


def reset_stream_manager() -> None:
    """Reset the global stream manager (for testing)."""
    global _stream_manager
    _stream_manager = None


# ============================================================================
# MODULE DEMO
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 70)
    print("  VØX Streaming Infrastructure Demo")
    print("=" * 70)

    # Test sentence segmentation
    print("\n1. Testing SentenceSegmenter...")
    test_text = "Hello world. How are you today? I am doing well! This is AXIOM VØX."
    sentences = SentenceSegmenter.segment(test_text)
    print(f"   Input: {test_text}")
    print(f"   Sentences: {sentences}")
    print(f"   Count: {len(sentences)}")

    # Test with abbreviations
    print("\n2. Testing abbreviation handling...")
    abbrev_text = "Dr. Smith said it was ok. Mrs. Jones agreed."
    abbrev_sentences = SentenceSegmenter.segment(abbrev_text)
    print(f"   Input: {abbrev_text}")
    print(f"   Sentences: {abbrev_sentences}")

    # Test StreamConfig
    print("\n3. Testing StreamConfig...")
    config = StreamConfig()
    print(f"   Chunk size: {config.chunk_size_bytes} bytes")
    print(f"   Sample rate: {config.sample_rate} Hz")
    print(f"   First chunk target: {config.max_first_chunk_ms}ms")

    # Test AudioChunk
    print("\n4. Testing AudioChunk...")
    chunk = AudioChunk(
        data=b'\x00' * 4096,
        index=0,
        timestamp_ms=100.0,
        duration_ms=170.0,
        is_sentence_end=True,
        sentence_index=0,
    )
    print(f"   Chunk: {chunk.to_dict()}")

    # Test StreamMessage
    print("\n5. Testing StreamMessage...")
    msg = StreamMessage(
        type=MessageType.STARTED,
        request_id="test_123",
        metadata={"total_sentences": 4},
    )
    print(f"   Message: {msg.to_json()}")

    # Test StreamSession
    print("\n6. Testing StreamSession...")
    session = StreamSession(
        request_id="session_abc",
        text="Hello world.",
        voice_id="default",
    )
    session.total_sentences = 1
    session.started_at = time.time()
    session.first_chunk_at = time.time() + 0.3
    print(f"   Session state: {session.state.value}")
    print(f"   Progress: {session.get_progress()}")

    # Test placeholder streaming
    print("\n7. Testing placeholder streaming...")

    async def test_placeholder():
        try:
            from axiom_vox.synthesis import VoxSynthesizer

            synth = VoxSynthesizer()
            streaming_synth = StreamingSynthesizer(synth, StreamConfig())

            text = "Hello world. This is a test."
            chunk_count = 0

            async for chunk in streaming_synth.synthesize_stream_placeholder(text, "default"):
                chunk_count += 1
                if chunk_count == 1:
                    print(f"   First chunk at {chunk.timestamp_ms:.1f}ms")
                if chunk.is_sentence_end:
                    print(f"   Sentence {chunk.sentence_index} complete")

            print(f"   Total chunks: {chunk_count}")

        except ImportError as e:
            print(f"   Skipped (missing dependencies): {e}")

    asyncio.run(test_placeholder())

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
