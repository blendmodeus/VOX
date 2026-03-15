"""
VØX STT Streaming
-----------------

Real-time streaming transcription for live audio input.
Accepts audio chunks and yields partial transcripts as they become available.

This replaces the current "record → stop → send blob" flow with
continuous recognition: audio streams in, text streams out.

Usage:
    from axiom_vox.stt import StreamingTranscriber, STTStreamConfig

    config = STTStreamConfig(chunk_duration_ms=500)
    streamer = StreamingTranscriber(config)

    session = streamer.create_session()
    for partial in streamer.feed_chunk(session.session_id, audio_chunk):
        print(f"Partial: {partial.text}")
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class STTStreamState(str, Enum):
    """State of a streaming transcription session."""
    PENDING = "pending"
    LISTENING = "listening"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class STTStreamConfig:
    """Configuration for streaming transcription."""
    chunk_duration_ms: int = 500
    sample_rate: int = 16000
    channels: int = 1
    vad_threshold: float = 0.5
    silence_timeout_ms: int = 2000  # End after this much silence
    max_duration_ms: int = 120000  # 2 minutes max
    language: Optional[str] = None
    model_size: str = "base"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_duration_ms": self.chunk_duration_ms,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "vad_threshold": self.vad_threshold,
            "silence_timeout_ms": self.silence_timeout_ms,
            "max_duration_ms": self.max_duration_ms,
            "language": self.language,
            "model_size": self.model_size,
        }


@dataclass
class PartialTranscript:
    """A partial transcription result from streaming."""
    text: str
    is_final: bool = False
    confidence: float = 0.0
    timestamp_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "is_final": self.is_final,
            "confidence": round(self.confidence, 3),
            "timestamp_ms": round(self.timestamp_ms, 1),
        }


@dataclass
class STTStreamSession:
    """A streaming transcription session."""
    session_id: str
    config: STTStreamConfig
    state: STTStreamState = STTStreamState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Audio tracking
    chunks_received: int = 0
    bytes_received: int = 0
    audio_duration_ms: float = 0.0

    # Transcription tracking
    partial_results: List[PartialTranscript] = field(default_factory=list)
    final_text: str = ""

    def get_progress(self) -> Dict[str, Any]:
        elapsed = 0.0
        if self.started_at:
            elapsed = (self.completed_at or time.time()) - self.started_at

        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "chunks_received": self.chunks_received,
            "bytes_received": self.bytes_received,
            "audio_duration_ms": round(self.audio_duration_ms, 1),
            "elapsed_s": round(elapsed, 2),
            "partials_count": len(self.partial_results),
            "final_text": self.final_text,
        }


@dataclass
class STTStreamMessage:
    """WebSocket protocol message for streaming STT."""
    type: str  # "start", "audio", "partial", "final", "error", "end"
    session_id: str
    data: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    error: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        msg = {
            "type": self.type,
            "session_id": self.session_id,
            "timestamp": time.time(),
        }
        if self.data:
            msg["data"] = self.data
        if self.text:
            msg["text"] = self.text
        if self.error:
            msg["error"] = self.error
        return msg


class StreamingTranscriber:
    """Real-time streaming speech-to-text.

    Manages sessions and processes audio chunks into partial transcripts.
    In the current implementation, this buffers audio and uses
    faster-whisper for final transcription. Future versions will
    support true streaming with Whisper's streaming mode.
    """

    def __init__(self, config: Optional[STTStreamConfig] = None):
        self.config = config or STTStreamConfig()
        self._sessions: Dict[str, STTStreamSession] = {}
        self._audio_buffers: Dict[str, bytearray] = {}

    def create_session(
        self,
        config: Optional[STTStreamConfig] = None,
    ) -> STTStreamSession:
        """Create a new streaming session."""
        session_config = config or self.config
        session = STTStreamSession(
            session_id=f"stt_stream_{uuid.uuid4().hex[:12]}",
            config=session_config,
        )
        self._sessions[session.session_id] = session
        self._audio_buffers[session.session_id] = bytearray()
        logger.info(f"Created STT stream session: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[STTStreamSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def feed_chunk(
        self,
        session_id: str,
        audio_data: bytes,
    ) -> List[PartialTranscript]:
        """Feed an audio chunk to a streaming session.

        Returns any new partial transcripts generated.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Unknown session: {session_id}")

        if session.state == STTStreamState.PENDING:
            session.state = STTStreamState.LISTENING
            session.started_at = time.time()

        # Buffer the audio
        self._audio_buffers[session_id].extend(audio_data)
        session.chunks_received += 1
        session.bytes_received += len(audio_data)
        session.audio_duration_ms += self.config.chunk_duration_ms

        # For now, return empty partials — full transcription happens on end_session
        # Future: implement VAD-triggered partial transcription
        return []

    def end_session(self, session_id: str) -> Optional[str]:
        """End a session and get the final transcription.

        Sends buffered audio to faster-whisper for transcription.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        session.state = STTStreamState.PROCESSING
        audio_buffer = self._audio_buffers.get(session_id, bytearray())

        if not audio_buffer:
            session.state = STTStreamState.COMPLETED
            session.completed_at = time.time()
            return ""

        try:
            from axiom_vox.stt.transcriber import get_transcriber
            from axiom_vox.stt.models import TranscriptionConfig

            config = TranscriptionConfig(
                model_size=session.config.model_size,
                language=session.config.language,
            )

            transcriber = get_transcriber()
            result = transcriber.transcribe_bytes(bytes(audio_buffer), config)

            session.final_text = result.text
            session.state = STTStreamState.COMPLETED
            session.completed_at = time.time()

            return result.text

        except Exception as e:
            logger.error(f"Streaming transcription error: {e}")
            session.state = STTStreamState.ERROR
            session.completed_at = time.time()
            return None

    def cancel_session(self, session_id: str) -> bool:
        """Cancel a streaming session."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        session.state = STTStreamState.CANCELLED
        session.completed_at = time.time()
        self._cleanup_session(session_id)
        return True

    def _cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        self._audio_buffers.pop(session_id, None)

    def remove_session(self, session_id: str) -> None:
        """Remove a session entirely."""
        self._sessions.pop(session_id, None)
        self._cleanup_session(session_id)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_default_streaming: Optional[StreamingTranscriber] = None


def get_streaming_transcriber() -> StreamingTranscriber:
    """Get or create the default streaming transcriber."""
    global _default_streaming
    if _default_streaming is None:
        _default_streaming = StreamingTranscriber()
    return _default_streaming
