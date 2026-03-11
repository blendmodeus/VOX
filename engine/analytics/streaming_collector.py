"""
AXIOM VOX Streaming Analytics Collector
---------------------------------------

Real-time metrics collection during streaming TTS synthesis.

Features:
- Per-chunk metrics (lightweight, ~0.5ms overhead)
- Cumulative session statistics (running averages)
- Event detection (late chunks, stutters, quality drops)
- Memory-efficient ring buffer for recent chunks
- Thread-safe for concurrent streams

Usage:
    from axiom_vox.analytics import StreamingAnalyticsCollector

    collector = StreamingAnalyticsCollector()

    # Start session
    collector.start_session("stream_123", "voice_1")

    # Record each chunk (called from StreamManager.stream())
    for chunk in stream:
        collector.record_chunk(
            "stream_123",
            chunk_index=chunk.index,
            chunk_data=chunk.data,
            timestamp_ms=chunk.timestamp_ms,
            duration_ms=chunk.duration_ms,
        )

    # End session and get final metrics
    metrics = collector.end_session("stream_123")
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class StreamingEventType(str, Enum):
    """Streaming-specific event types."""
    CHUNK_LATE = "chunk_late"
    STUTTER_DETECTED = "stutter_detected"
    QUALITY_DROP = "quality_drop"
    FIRST_CHUNK = "first_chunk"
    SENTENCE_BOUNDARY = "sentence_boundary"
    STREAM_DEGRADED = "stream_degraded"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class StreamChunkMetrics:
    """
    Per-chunk metrics (lightweight, computed inline).

    Captures timing and optional quality data for a single audio chunk.
    """
    chunk_index: int
    timestamp_ms: float  # Wall clock since stream start
    latency_ms: float  # Time since previous chunk
    size_bytes: int
    duration_ms: float  # Audio duration in chunk
    is_sentence_end: bool
    sentence_index: Optional[int] = None

    # Lightweight quality (optional, ~1ms if enabled)
    peak_amplitude: Optional[float] = None
    rms_level: Optional[float] = None

    # Timing
    expected_latency_ms: float = 0.0  # Based on previous chunk duration
    is_late: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_index": self.chunk_index,
            "timestamp_ms": self.timestamp_ms,
            "latency_ms": self.latency_ms,
            "size_bytes": self.size_bytes,
            "duration_ms": self.duration_ms,
            "is_sentence_end": self.is_sentence_end,
            "sentence_index": self.sentence_index,
            "peak_amplitude": self.peak_amplitude,
            "rms_level": self.rms_level,
            "expected_latency_ms": self.expected_latency_ms,
            "is_late": self.is_late,
        }


@dataclass
class StreamSessionMetrics:
    """
    Cumulative session statistics (updated per chunk).

    Maintains running statistics for a streaming session.
    """
    session_id: str
    voice_id: str

    # Timing
    started_at: float = 0.0
    first_chunk_at: Optional[float] = None
    last_chunk_at: Optional[float] = None

    # Counters
    total_chunks: int = 0
    total_bytes: int = 0
    total_audio_duration_ms: float = 0.0
    sentences_completed: int = 0

    # Latency statistics (running)
    latency_sum_ms: float = 0.0
    latency_sum_sq_ms: float = 0.0  # For variance calculation
    latency_min_ms: float = float("inf")
    latency_max_ms: float = 0.0
    late_chunks: int = 0

    # Quality (running)
    quality_sum: float = 0.0
    quality_samples: int = 0

    # Events
    stutters_detected: int = 0
    quality_drops: int = 0

    def update_latency(self, latency_ms: float, is_late: bool) -> None:
        """Update latency statistics with new chunk."""
        self.latency_sum_ms += latency_ms
        self.latency_sum_sq_ms += latency_ms ** 2
        self.latency_min_ms = min(self.latency_min_ms, latency_ms)
        self.latency_max_ms = max(self.latency_max_ms, latency_ms)
        if is_late:
            self.late_chunks += 1

    @property
    def avg_latency_ms(self) -> float:
        """Average inter-chunk latency."""
        if self.total_chunks <= 1:
            return 0.0
        return self.latency_sum_ms / (self.total_chunks - 1)

    @property
    def latency_std_ms(self) -> float:
        """Standard deviation of inter-chunk latency."""
        if self.total_chunks <= 2:
            return 0.0
        n = self.total_chunks - 1
        mean = self.avg_latency_ms
        variance = (self.latency_sum_sq_ms / n) - (mean ** 2)
        return max(0, variance) ** 0.5

    @property
    def late_chunk_ratio(self) -> float:
        """Ratio of chunks that arrived late."""
        if self.total_chunks <= 1:
            return 0.0
        return self.late_chunks / (self.total_chunks - 1)

    @property
    def first_chunk_latency_ms(self) -> Optional[float]:
        """Time from session start to first chunk."""
        if self.first_chunk_at and self.started_at:
            return (self.first_chunk_at - self.started_at) * 1000
        return None

    @property
    def real_time_factor(self) -> float:
        """Streaming RTF (elapsed time / audio duration)."""
        if self.total_audio_duration_ms == 0:
            return 0.0
        elapsed = (self.last_chunk_at or time.time()) - self.started_at
        return (elapsed * 1000) / self.total_audio_duration_ms

    def get_current_quality_score(self) -> float:
        """Current running quality score."""
        if self.quality_samples == 0:
            return 1.0  # Assume good if no quality data
        return self.quality_sum / self.quality_samples

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/storage."""
        return {
            "session_id": self.session_id,
            "voice_id": self.voice_id,
            "started_at": self.started_at,
            "total_chunks": self.total_chunks,
            "total_bytes": self.total_bytes,
            "total_audio_duration_ms": self.total_audio_duration_ms,
            "sentences_completed": self.sentences_completed,
            "first_chunk_latency_ms": self.first_chunk_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "latency_std_ms": self.latency_std_ms,
            "latency_min_ms": self.latency_min_ms if self.latency_min_ms != float("inf") else None,
            "latency_max_ms": self.latency_max_ms,
            "late_chunk_ratio": self.late_chunk_ratio,
            "late_chunks": self.late_chunks,
            "real_time_factor": self.real_time_factor,
            "current_quality_score": self.get_current_quality_score(),
            "stutters_detected": self.stutters_detected,
            "quality_drops": self.quality_drops,
        }


@dataclass
class StreamingEvent:
    """Real-time streaming event for monitoring."""
    event_type: StreamingEventType
    session_id: str
    timestamp: float
    chunk_index: Optional[int] = None
    severity: str = "info"  # info, warning, critical
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "chunk_index": self.chunk_index,
            "severity": self.severity,
            "message": self.message,
            "data": self.data,
        }


# ============================================================================
# STREAMING ANALYTICS COLLECTOR
# ============================================================================


class StreamingAnalyticsCollector:
    """
    Real-time metrics collector for streaming synthesis.

    Thread-safe collector that tracks per-chunk metrics, cumulative
    session statistics, and detects streaming issues like stutters
    and late chunks.

    Usage:
        collector = StreamingAnalyticsCollector()

        # Start session
        collector.start_session("stream_123", "voice_1")

        # Record each chunk
        for chunk in stream:
            collector.record_chunk("stream_123", chunk.index, ...)

        # End session
        metrics = collector.end_session("stream_123")
    """

    def __init__(
        self,
        late_threshold_ms: float = 200.0,  # Chunk late if > expected + threshold
        stutter_threshold_ms: float = 500.0,  # Gap > threshold = stutter
        quality_drop_threshold: float = 0.3,  # Alert if quality drops by this much
        max_recent_chunks: int = 100,  # Ring buffer size
        enable_lightweight_quality: bool = False,  # Compute peak/RMS per chunk
        event_callback: Optional[Callable[[StreamingEvent], None]] = None,
    ):
        """
        Initialize streaming analytics collector.

        Args:
            late_threshold_ms: Chunk considered late if latency > expected + threshold
            stutter_threshold_ms: Gap > threshold triggers stutter detection
            quality_drop_threshold: Quality change threshold for alerts (0-1)
            max_recent_chunks: Maximum chunks to keep in ring buffer per session
            enable_lightweight_quality: Compute peak/RMS metrics per chunk
            event_callback: Function called when events are detected
        """
        self.late_threshold_ms = late_threshold_ms
        self.stutter_threshold_ms = stutter_threshold_ms
        self.quality_drop_threshold = quality_drop_threshold
        self.max_recent_chunks = max_recent_chunks
        self.enable_lightweight_quality = enable_lightweight_quality
        self.event_callback = event_callback

        self._sessions: Dict[str, StreamSessionMetrics] = {}
        self._recent_chunks: Dict[str, deque] = {}  # Ring buffer per session
        self._last_chunk_time: Dict[str, float] = {}
        self._last_quality: Dict[str, float] = {}
        self._lock = threading.Lock()

    def start_session(
        self,
        session_id: str,
        voice_id: str,
    ) -> StreamSessionMetrics:
        """
        Start tracking a streaming session.

        Args:
            session_id: Unique session identifier
            voice_id: Voice ID being used

        Returns:
            StreamSessionMetrics instance for this session
        """
        with self._lock:
            metrics = StreamSessionMetrics(
                session_id=session_id,
                voice_id=voice_id,
                started_at=time.time(),
            )
            self._sessions[session_id] = metrics
            self._recent_chunks[session_id] = deque(maxlen=self.max_recent_chunks)
            self._last_chunk_time[session_id] = metrics.started_at
            self._last_quality[session_id] = 1.0

            logger.debug(f"Started streaming analytics for session: {session_id}")
            return metrics

    def record_chunk(
        self,
        session_id: str,
        chunk_index: int,
        chunk_data: bytes,
        timestamp_ms: float,
        duration_ms: float,
        is_sentence_end: bool = False,
        sentence_index: Optional[int] = None,
        sample_rate: int = 24000,
    ) -> Optional[StreamChunkMetrics]:
        """
        Record metrics for a single chunk.

        Called from StreamManager.stream() for each yielded chunk.

        Args:
            session_id: Session identifier
            chunk_index: Index of this chunk in the stream
            chunk_data: Raw audio bytes
            timestamp_ms: Timestamp from stream start (chunk.timestamp_ms)
            duration_ms: Audio duration in this chunk
            is_sentence_end: Whether this chunk ends a sentence
            sentence_index: Which sentence this ends (if is_sentence_end)
            sample_rate: Audio sample rate

        Returns:
            StreamChunkMetrics for this chunk, or None if session not found
        """
        now = time.time()

        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            # Compute inter-chunk latency
            last_time = self._last_chunk_time.get(session_id, session.started_at)
            latency_ms = (now - last_time) * 1000

            # Expected latency based on previous chunk duration
            # (ideally chunks arrive just as previous finishes playing)
            expected_latency_ms = duration_ms if chunk_index > 0 else 0
            is_late = latency_ms > (expected_latency_ms + self.late_threshold_ms)

            # Lightweight quality metrics (optional)
            peak_amplitude = None
            rms_level = None
            if self.enable_lightweight_quality and len(chunk_data) > 0:
                peak_amplitude, rms_level = self._compute_lightweight_quality(
                    chunk_data, sample_rate
                )

            # Create chunk metrics
            chunk_metrics = StreamChunkMetrics(
                chunk_index=chunk_index,
                timestamp_ms=timestamp_ms,
                latency_ms=latency_ms,
                size_bytes=len(chunk_data),
                duration_ms=duration_ms,
                is_sentence_end=is_sentence_end,
                sentence_index=sentence_index,
                peak_amplitude=peak_amplitude,
                rms_level=rms_level,
                expected_latency_ms=expected_latency_ms,
                is_late=is_late,
            )

            # Update session metrics
            session.total_chunks += 1
            session.total_bytes += len(chunk_data)
            session.total_audio_duration_ms += duration_ms
            session.last_chunk_at = now

            if chunk_index == 0:
                session.first_chunk_at = now
                self._emit_event(StreamingEvent(
                    event_type=StreamingEventType.FIRST_CHUNK,
                    session_id=session_id,
                    timestamp=now,
                    chunk_index=0,
                    severity="info",
                    message=f"First chunk latency: {(now - session.started_at) * 1000:.1f}ms",
                    data={"first_chunk_latency_ms": (now - session.started_at) * 1000},
                ))

            if chunk_index > 0:
                session.update_latency(latency_ms, is_late)

            if is_sentence_end:
                session.sentences_completed += 1
                self._emit_event(StreamingEvent(
                    event_type=StreamingEventType.SENTENCE_BOUNDARY,
                    session_id=session_id,
                    timestamp=now,
                    chunk_index=chunk_index,
                    severity="info",
                    message=f"Sentence {sentence_index} completed",
                    data={"sentence_index": sentence_index},
                ))

            # Store in ring buffer
            self._recent_chunks[session_id].append(chunk_metrics)
            self._last_chunk_time[session_id] = now

            # Detect events
            self._detect_events(session_id, chunk_metrics, session)

            return chunk_metrics

    def get_current_metrics(self, session_id: str) -> Optional[StreamSessionMetrics]:
        """
        Get current cumulative metrics for a session.

        Args:
            session_id: Session identifier

        Returns:
            StreamSessionMetrics or None if session not found
        """
        with self._lock:
            return self._sessions.get(session_id)

    def get_recent_chunks(
        self,
        session_id: str,
        count: int = 10,
    ) -> List[StreamChunkMetrics]:
        """
        Get recent chunk metrics from ring buffer.

        Args:
            session_id: Session identifier
            count: Maximum number of recent chunks to return

        Returns:
            List of recent StreamChunkMetrics
        """
        with self._lock:
            buffer = self._recent_chunks.get(session_id)
            if not buffer:
                return []
            return list(buffer)[-count:]

    def end_session(self, session_id: str) -> Optional[StreamSessionMetrics]:
        """
        End a streaming session and return final metrics.

        Cleans up internal state but returns metrics for storage.

        Args:
            session_id: Session identifier

        Returns:
            Final StreamSessionMetrics, or None if session not found
        """
        with self._lock:
            metrics = self._sessions.pop(session_id, None)
            self._recent_chunks.pop(session_id, None)
            self._last_chunk_time.pop(session_id, None)
            self._last_quality.pop(session_id, None)

            if metrics:
                logger.debug(
                    f"Ended streaming analytics for session: {session_id}, "
                    f"chunks={metrics.total_chunks}, rtf={metrics.real_time_factor:.3f}"
                )

            return metrics

    def get_active_session_count(self) -> int:
        """Get count of active streaming sessions."""
        with self._lock:
            return len(self._sessions)

    def get_active_session_ids(self) -> List[str]:
        """Get list of active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    def _compute_lightweight_quality(
        self,
        chunk_data: bytes,
        sample_rate: int,
    ) -> tuple:
        """
        Compute lightweight quality metrics (~0.5ms).

        Args:
            chunk_data: Raw audio bytes
            sample_rate: Audio sample rate

        Returns:
            Tuple of (peak_amplitude, rms_level) or (None, None) on error
        """
        try:
            import numpy as np
            # Assume 32-bit float audio
            samples = np.frombuffer(chunk_data, dtype=np.float32)
            if len(samples) == 0:
                return None, None
            peak = float(np.max(np.abs(samples)))
            rms = float(np.sqrt(np.mean(samples ** 2)))
            return peak, rms
        except Exception:
            return None, None

    def _detect_events(
        self,
        session_id: str,
        chunk: StreamChunkMetrics,
        session: StreamSessionMetrics,
    ) -> None:
        """
        Detect and emit streaming events.

        Args:
            session_id: Session identifier
            chunk: Current chunk metrics
            session: Session metrics
        """
        now = time.time()

        # Late chunk detection
        if chunk.is_late:
            overage_ms = chunk.latency_ms - chunk.expected_latency_ms
            severity = "warning" if chunk.latency_ms < self.stutter_threshold_ms else "critical"
            self._emit_event(StreamingEvent(
                event_type=StreamingEventType.CHUNK_LATE,
                session_id=session_id,
                timestamp=now,
                chunk_index=chunk.chunk_index,
                severity=severity,
                message=f"Chunk {chunk.chunk_index} late by {overage_ms:.1f}ms",
                data={
                    "latency_ms": chunk.latency_ms,
                    "expected_ms": chunk.expected_latency_ms,
                    "overage_ms": overage_ms,
                },
            ))

        # Stutter detection (large gap)
        if chunk.latency_ms > self.stutter_threshold_ms and chunk.chunk_index > 0:
            session.stutters_detected += 1
            self._emit_event(StreamingEvent(
                event_type=StreamingEventType.STUTTER_DETECTED,
                session_id=session_id,
                timestamp=now,
                chunk_index=chunk.chunk_index,
                severity="warning",
                message=f"Stutter detected: {chunk.latency_ms:.1f}ms gap",
                data={"gap_ms": chunk.latency_ms},
            ))

        # Quality drop detection (if lightweight quality enabled)
        if chunk.rms_level is not None:
            # Simple quality heuristic: RMS should be > 0.01 for speech
            current_quality = min(1.0, chunk.rms_level / 0.1)
            last_quality = self._last_quality.get(session_id, 1.0)

            if last_quality - current_quality > self.quality_drop_threshold:
                session.quality_drops += 1
                self._emit_event(StreamingEvent(
                    event_type=StreamingEventType.QUALITY_DROP,
                    session_id=session_id,
                    timestamp=now,
                    chunk_index=chunk.chunk_index,
                    severity="warning",
                    message=f"Quality dropped from {last_quality:.2f} to {current_quality:.2f}",
                    data={
                        "previous_quality": last_quality,
                        "current_quality": current_quality,
                    },
                ))

            self._last_quality[session_id] = current_quality
            session.quality_sum += current_quality
            session.quality_samples += 1

    def _emit_event(self, event: StreamingEvent) -> None:
        """
        Emit event to callback if registered.

        Args:
            event: StreamingEvent to emit
        """
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
        else:
            # Log event if no callback
            if event.severity in ("warning", "critical"):
                logger.warning(f"Streaming event: {event.event_type.value} - {event.message}")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_streaming_collector: Optional[StreamingAnalyticsCollector] = None


def get_streaming_collector() -> StreamingAnalyticsCollector:
    """Get or create the global streaming analytics collector."""
    global _streaming_collector
    if _streaming_collector is None:
        _streaming_collector = StreamingAnalyticsCollector()
    return _streaming_collector


def set_streaming_collector(collector: StreamingAnalyticsCollector) -> None:
    """Set the global streaming analytics collector."""
    global _streaming_collector
    _streaming_collector = collector


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  AXIOM VOX Streaming Analytics Collector Demo")
    print("=" * 70)

    events: List[StreamingEvent] = []

    def event_handler(event: StreamingEvent):
        events.append(event)
        print(f"  EVENT: {event.event_type.value} - {event.message}")

    collector = StreamingAnalyticsCollector(
        late_threshold_ms=50.0,
        stutter_threshold_ms=200.0,
        event_callback=event_handler,
    )

    # Simulate a streaming session
    session_id = "demo_stream_001"
    print(f"\nStarting session: {session_id}")

    collector.start_session(session_id, "demo_voice")

    # Simulate 10 chunks with varying latencies
    for i in range(10):
        # Simulate some delay
        if i == 5:
            time.sleep(0.3)  # Stutter on chunk 5
        else:
            time.sleep(0.05)  # Normal delay

        chunk = collector.record_chunk(
            session_id=session_id,
            chunk_index=i,
            chunk_data=b"\x00" * 4096,
            timestamp_ms=i * 100.0,
            duration_ms=100.0,
            is_sentence_end=(i == 4 or i == 9),
            sentence_index=(0 if i == 4 else 1 if i == 9 else None),
        )

        if chunk:
            print(f"  Chunk {i}: latency={chunk.latency_ms:.1f}ms, late={chunk.is_late}")

    # Get final metrics
    metrics = collector.end_session(session_id)

    print("\n" + "-" * 70)
    print("Final Session Metrics:")
    print("-" * 70)
    for key, value in metrics.to_dict().items():
        if value is not None:
            print(f"  {key}: {value}")

    print(f"\nTotal events emitted: {len(events)}")
    print("=" * 70)
