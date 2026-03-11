"""
Tests for AXIOM VOX Streaming Analytics
----------------------------------------

Comprehensive tests for real-time streaming analytics:
- StreamChunkMetrics and StreamSessionMetrics data models
- StreamingAnalyticsCollector lifecycle and event detection
- Ring buffer behavior for recent chunks
- Storage integration for persistence
- Integration with streaming pipeline

Run with: pytest axiom_vox/tests/test_streaming_analytics.py -v
"""

import time
import threading
import pytest
from typing import List

from axiom_vox.analytics.streaming_collector import (
    StreamChunkMetrics,
    StreamSessionMetrics,
    StreamingEvent,
    StreamingEventType,
    StreamingAnalyticsCollector,
    get_streaming_collector,
    set_streaming_collector,
)
from axiom_vox.analytics.models import (
    StreamingSessionAnalytics,
    AnalyticsEventType,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def collector():
    """Create a fresh StreamingAnalyticsCollector for each test."""
    return StreamingAnalyticsCollector(
        late_threshold_ms=200.0,
        stutter_threshold_ms=500.0,
        quality_drop_threshold=0.3,
        max_recent_chunks=50,
        enable_lightweight_quality=True,
    )


@pytest.fixture
def events_captured():
    """Fixture to capture events from collector callback."""
    return []


@pytest.fixture
def collector_with_callback(events_captured):
    """Collector with event callback for testing event emission."""
    def callback(event: StreamingEvent):
        events_captured.append(event)

    return StreamingAnalyticsCollector(
        late_threshold_ms=200.0,
        stutter_threshold_ms=500.0,
        event_callback=callback,
    )


# ============================================================================
# STREAM CHUNK METRICS TESTS
# ============================================================================

class TestStreamChunkMetrics:
    """Tests for StreamChunkMetrics data model."""

    def test_basic_creation(self):
        """Test basic chunk metrics creation."""
        metrics = StreamChunkMetrics(
            chunk_index=0,
            timestamp_ms=100.0,
            latency_ms=50.0,
            size_bytes=4096,
            duration_ms=100.0,
            is_sentence_end=False,
            is_late=False,
        )

        assert metrics.chunk_index == 0
        assert metrics.timestamp_ms == 100.0
        assert metrics.latency_ms == 50.0
        assert metrics.size_bytes == 4096
        assert metrics.duration_ms == 100.0
        assert not metrics.is_sentence_end
        assert not metrics.is_late
        assert metrics.peak_amplitude is None

    def test_late_chunk_detection(self):
        """Test late chunk flag."""
        late_chunk = StreamChunkMetrics(
            chunk_index=5,
            timestamp_ms=600.0,
            latency_ms=350.0,  # Exceeds typical threshold
            size_bytes=4096,
            duration_ms=100.0,
            is_sentence_end=False,
            is_late=True,
        )

        assert late_chunk.is_late

    def test_with_peak_amplitude(self):
        """Test chunk metrics with lightweight quality metric."""
        metrics = StreamChunkMetrics(
            chunk_index=0,
            timestamp_ms=100.0,
            latency_ms=50.0,
            size_bytes=4096,
            duration_ms=100.0,
            is_sentence_end=False,
            is_late=False,
            peak_amplitude=0.75,
        )

        assert metrics.peak_amplitude == 0.75

    def test_to_dict(self):
        """Test dictionary serialization."""
        metrics = StreamChunkMetrics(
            chunk_index=3,
            timestamp_ms=400.0,
            latency_ms=95.0,
            size_bytes=2048,
            duration_ms=50.0,
            is_sentence_end=True,
            is_late=False,
            peak_amplitude=0.5,
        )

        d = metrics.to_dict()

        assert d["chunk_index"] == 3
        assert d["timestamp_ms"] == 400.0
        assert d["latency_ms"] == 95.0
        assert d["size_bytes"] == 2048
        assert d["duration_ms"] == 50.0
        assert d["is_sentence_end"] is True
        assert d["is_late"] is False
        assert d["peak_amplitude"] == 0.5


# ============================================================================
# STREAM SESSION METRICS TESTS
# ============================================================================

class TestStreamSessionMetrics:
    """Tests for StreamSessionMetrics data model."""

    def test_basic_creation(self):
        """Test basic session metrics creation."""
        metrics = StreamSessionMetrics(
            session_id="test_session",
            voice_id="voice_1",
            started_at=time.time(),
        )

        assert metrics.session_id == "test_session"
        assert metrics.voice_id == "voice_1"
        assert metrics.total_chunks == 0
        assert metrics.total_bytes == 0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.stutters_detected == 0

    def test_update_with_chunk(self):
        """Test metrics update with chunk data."""
        metrics = StreamSessionMetrics(
            session_id="test_session",
            voice_id="voice_1",
            started_at=time.time(),
        )

        # Simulate adding a chunk
        metrics.total_chunks = 1
        metrics.total_bytes = 4096
        metrics.total_audio_duration_ms = 100.0

        assert metrics.total_chunks == 1
        assert metrics.total_bytes == 4096

    def test_latency_statistics(self):
        """Test latency statistics tracking."""
        metrics = StreamSessionMetrics(
            session_id="test",
            voice_id="voice_1",
            started_at=time.time(),
        )

        # Simulate multiple chunks with varying latency
        latencies = [50.0, 75.0, 100.0, 60.0, 80.0]
        metrics.total_chunks = len(latencies) + 1  # +1 for first chunk

        for latency in latencies:
            metrics.update_latency(latency, is_late=False)

        assert metrics.avg_latency_ms > 0
        assert metrics.latency_min_ms == 50.0
        assert metrics.latency_max_ms == 100.0
        assert metrics.latency_std_ms > 0

    def test_to_dict(self):
        """Test dictionary serialization."""
        metrics = StreamSessionMetrics(
            session_id="test",
            voice_id="voice_1",
            started_at=1000.0,
        )
        metrics.total_chunks = 10
        metrics.stutters_detected = 1

        d = metrics.to_dict()

        assert d["session_id"] == "test"
        assert d["voice_id"] == "voice_1"
        assert d["total_chunks"] == 10
        assert d["stutters_detected"] == 1


# ============================================================================
# STREAMING EVENT TESTS
# ============================================================================

class TestStreamingEvent:
    """Tests for StreamingEvent data model."""

    def test_chunk_late_event(self):
        """Test late chunk event creation."""
        event = StreamingEvent(
            event_type=StreamingEventType.CHUNK_LATE,
            session_id="test_session",
            chunk_index=5,
            timestamp=time.time(),
            message="Chunk 5 late by 150ms",
            data={"latency_ms": 350.0, "threshold_ms": 200.0},
        )

        assert event.event_type == StreamingEventType.CHUNK_LATE
        assert event.chunk_index == 5
        assert "late" in event.message.lower()

    def test_stutter_event(self):
        """Test stutter detected event."""
        event = StreamingEvent(
            event_type=StreamingEventType.STUTTER_DETECTED,
            session_id="test_session",
            chunk_index=10,
            timestamp=time.time(),
            message="Stutter detected: 600ms gap",
            data={"gap_ms": 600.0},
        )

        assert event.event_type == StreamingEventType.STUTTER_DETECTED

    def test_to_dict(self):
        """Test event serialization."""
        event = StreamingEvent(
            event_type=StreamingEventType.FIRST_CHUNK,
            session_id="test",
            chunk_index=0,
            timestamp=1000.0,
            message="First chunk received",
            data={"latency_ms": 200.0},
        )

        d = event.to_dict()

        assert d["event_type"] == "first_chunk"
        assert d["session_id"] == "test"
        assert d["chunk_index"] == 0


# ============================================================================
# STREAMING ANALYTICS COLLECTOR TESTS
# ============================================================================

class TestStreamingAnalyticsCollector:
    """Tests for StreamingAnalyticsCollector."""

    def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector.late_threshold_ms == 200.0
        assert collector.stutter_threshold_ms == 500.0
        assert collector.max_recent_chunks == 50

    def test_start_session(self, collector):
        """Test starting a streaming session."""
        metrics = collector.start_session("session_1", "voice_1")

        assert metrics is not None
        assert metrics.session_id == "session_1"
        assert metrics.voice_id == "voice_1"
        assert metrics.total_chunks == 0

    def test_start_duplicate_session(self, collector):
        """Test starting a session that already exists."""
        collector.start_session("session_1", "voice_1")

        # Starting again should return a new session (overwrites)
        metrics = collector.start_session("session_1", "voice_1")

        assert metrics.session_id == "session_1"

    def test_record_chunk_basic(self, collector):
        """Test recording a basic chunk."""
        collector.start_session("session_1", "voice_1")

        chunk = collector.record_chunk(
            session_id="session_1",
            chunk_index=0,
            chunk_data=b"\x00" * 4096,
            timestamp_ms=100.0,
            duration_ms=100.0,
            is_sentence_end=False,
        )

        assert chunk is not None
        assert chunk.chunk_index == 0
        assert chunk.size_bytes == 4096
        assert chunk.duration_ms == 100.0

    def test_record_multiple_chunks(self, collector):
        """Test recording multiple chunks."""
        collector.start_session("session_1", "voice_1")

        for i in range(10):
            collector.record_chunk(
                session_id="session_1",
                chunk_index=i,
                chunk_data=b"\x00" * 4096,
                timestamp_ms=i * 100.0,
                duration_ms=100.0,
                is_sentence_end=(i % 5 == 4),
            )

        metrics = collector.get_current_metrics("session_1")

        assert metrics.total_chunks == 10
        assert metrics.total_bytes == 4096 * 10
        assert metrics.total_audio_duration_ms == 1000.0

    def test_record_chunk_without_session(self, collector):
        """Test recording chunk for non-existent session."""
        chunk = collector.record_chunk(
            session_id="nonexistent",
            chunk_index=0,
            chunk_data=b"\x00" * 4096,
            timestamp_ms=100.0,
            duration_ms=100.0,
        )

        assert chunk is None

    def test_late_chunk_detection(self, collector_with_callback, events_captured):
        """Test late chunk detection and event emission."""
        collector_with_callback.start_session("session_1", "voice_1")

        # First chunk - establishes baseline
        collector_with_callback.record_chunk(
            session_id="session_1",
            chunk_index=0,
            chunk_data=b"\x00" * 4096,
            timestamp_ms=0.0,
            duration_ms=100.0,
        )

        # Force a delay to trigger late detection
        time.sleep(0.35)  # 350ms delay - should be late (threshold 200ms)

        collector_with_callback.record_chunk(
            session_id="session_1",
            chunk_index=1,
            chunk_data=b"\x00" * 4096,
            timestamp_ms=100.0,
            duration_ms=100.0,
        )

        # Check for late chunk event
        late_events = [e for e in events_captured if e.event_type == StreamingEventType.CHUNK_LATE]
        assert len(late_events) >= 1

    def test_stutter_detection(self, collector_with_callback, events_captured):
        """Test stutter detection (gap > 500ms)."""
        collector_with_callback.start_session("session_1", "voice_1")

        collector_with_callback.record_chunk(
            session_id="session_1",
            chunk_index=0,
            chunk_data=b"\x00" * 4096,
            timestamp_ms=0.0,
            duration_ms=100.0,
        )

        # Force a stutter delay > 500ms
        time.sleep(0.55)

        collector_with_callback.record_chunk(
            session_id="session_1",
            chunk_index=1,
            chunk_data=b"\x00" * 4096,
            timestamp_ms=100.0,
            duration_ms=100.0,
        )

        stutter_events = [e for e in events_captured if e.event_type == StreamingEventType.STUTTER_DETECTED]
        assert len(stutter_events) >= 1

        metrics = collector_with_callback.get_current_metrics("session_1")
        assert metrics.stutters_detected >= 1

    def test_first_chunk_event(self, collector_with_callback, events_captured):
        """Test first chunk event emission."""
        collector_with_callback.start_session("session_1", "voice_1")

        collector_with_callback.record_chunk(
            session_id="session_1",
            chunk_index=0,
            chunk_data=b"\x00" * 4096,
            timestamp_ms=0.0,
            duration_ms=100.0,
        )

        first_events = [e for e in events_captured if e.event_type == StreamingEventType.FIRST_CHUNK]
        assert len(first_events) == 1

    def test_get_recent_chunks(self, collector):
        """Test retrieving recent chunks from ring buffer."""
        collector.start_session("session_1", "voice_1")

        for i in range(20):
            collector.record_chunk(
                session_id="session_1",
                chunk_index=i,
                chunk_data=b"\x00" * 1024,
                timestamp_ms=i * 100.0,
                duration_ms=100.0,
            )

        recent = collector.get_recent_chunks("session_1", count=5)

        assert len(recent) == 5
        # Should be most recent 5 chunks
        assert recent[-1].chunk_index == 19
        assert recent[-2].chunk_index == 18

    def test_ring_buffer_max_size(self, collector):
        """Test ring buffer doesn't exceed max size."""
        collector.start_session("session_1", "voice_1")

        # Record more chunks than max_recent_chunks (50)
        for i in range(100):
            collector.record_chunk(
                session_id="session_1",
                chunk_index=i,
                chunk_data=b"\x00" * 1024,
                timestamp_ms=i * 100.0,
                duration_ms=100.0,
            )

        recent = collector.get_recent_chunks("session_1", count=100)

        # Should only have max_recent_chunks (50)
        assert len(recent) <= 50
        # Most recent chunk should be index 99
        assert recent[-1].chunk_index == 99

    def test_end_session(self, collector):
        """Test ending a streaming session."""
        collector.start_session("session_1", "voice_1")

        for i in range(5):
            collector.record_chunk(
                session_id="session_1",
                chunk_index=i,
                chunk_data=b"\x00" * 4096,
                timestamp_ms=i * 100.0,
                duration_ms=100.0,
            )

        final_metrics = collector.end_session("session_1")

        assert final_metrics is not None
        assert final_metrics.total_chunks == 5
        assert final_metrics.last_chunk_at is not None

        # Session should be removed
        assert collector.get_current_metrics("session_1") is None

    def test_end_nonexistent_session(self, collector):
        """Test ending a session that doesn't exist."""
        result = collector.end_session("nonexistent")
        assert result is None

    def test_get_active_session_count(self, collector):
        """Test getting count of active sessions."""
        assert collector.get_active_session_count() == 0

        collector.start_session("session_1", "voice_1")
        assert collector.get_active_session_count() == 1

        collector.start_session("session_2", "voice_2")
        assert collector.get_active_session_count() == 2

        collector.end_session("session_1")
        assert collector.get_active_session_count() == 1

    def test_latency_statistics_calculation(self, collector):
        """Test accurate latency statistics."""
        collector.start_session("session_1", "voice_1")

        # Record chunks - first chunk doesn't contribute to inter-chunk latency
        for i in range(5):
            collector.record_chunk(
                session_id="session_1",
                chunk_index=i,
                chunk_data=b"\x00" * 1024,
                timestamp_ms=i * 100.0,
                duration_ms=50.0,
            )
            time.sleep(0.01)  # Small delay between chunks

        metrics = collector.get_current_metrics("session_1")

        assert metrics.total_chunks == 5
        assert metrics.latency_min_ms > 0
        assert metrics.latency_max_ms > 0
        assert metrics.avg_latency_ms > 0

    def test_real_time_factor_calculation(self, collector):
        """Test real-time factor calculation."""
        collector.start_session("session_1", "voice_1")

        for i in range(10):
            collector.record_chunk(
                session_id="session_1",
                chunk_index=i,
                chunk_data=b"\x00" * 4096,
                timestamp_ms=i * 50.0,
                duration_ms=100.0,  # 100ms audio per chunk
            )
            time.sleep(0.01)

        metrics = collector.get_current_metrics("session_1")

        # real_time_factor = elapsed_time / audio_duration
        assert metrics.real_time_factor >= 0


# ============================================================================
# GLOBAL COLLECTOR TESTS
# ============================================================================

class TestGlobalCollector:
    """Tests for global collector singleton."""

    def test_get_default_collector(self):
        """Test getting default global collector."""
        collector = get_streaming_collector()
        assert collector is not None
        assert isinstance(collector, StreamingAnalyticsCollector)

    def test_set_custom_collector(self):
        """Test setting custom global collector."""
        # Save original
        original = get_streaming_collector()

        custom = StreamingAnalyticsCollector(late_threshold_ms=100.0)
        set_streaming_collector(custom)

        assert get_streaming_collector() is custom

        # Restore original
        set_streaming_collector(original)


# ============================================================================
# STREAMING SESSION ANALYTICS TESTS
# ============================================================================

class TestStreamingSessionAnalytics:
    """Tests for StreamingSessionAnalytics model."""

    def test_creation(self):
        """Test model creation."""
        analytics = StreamingSessionAnalytics(
            session_id="test_session",
            voice_id="voice_1",
            timestamp=time.time(),
            text_length=100,
            total_chunks=10,
            first_chunk_latency_ms=200.0,
            avg_chunk_latency_ms=80.0,
            stutters_detected=0,
        )

        assert analytics.session_id == "test_session"
        assert analytics.total_chunks == 10
        assert analytics.first_chunk_latency_ms == 200.0

    def test_to_dict(self):
        """Test dictionary serialization."""
        analytics = StreamingSessionAnalytics(
            session_id="test",
            voice_id="voice_1",
            timestamp=1000.0,
            text_length=50,
        )

        d = analytics.to_dict()

        assert d["session_id"] == "test"
        assert d["voice_id"] == "voice_1"
        assert d["timestamp"] == 1000.0


# ============================================================================
# THREAD SAFETY TESTS
# ============================================================================

class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_concurrent_chunk_recording(self, collector):
        """Test concurrent chunk recording from multiple threads."""
        collector.start_session("session_1", "voice_1")

        errors = []

        def record_chunks(start_index: int, count: int):
            try:
                for i in range(count):
                    collector.record_chunk(
                        session_id="session_1",
                        chunk_index=start_index + i,
                        chunk_data=b"\x00" * 1024,
                        timestamp_ms=(start_index + i) * 10.0,
                        duration_ms=10.0,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_chunks, args=(i * 100, 50))
            for i in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        metrics = collector.get_current_metrics("session_1")
        assert metrics.total_chunks == 200  # 4 threads * 50 chunks

    def test_concurrent_sessions(self, collector):
        """Test multiple concurrent sessions."""
        def run_session(session_id: str, voice_id: str):
            collector.start_session(session_id, voice_id)
            for i in range(10):
                collector.record_chunk(
                    session_id=session_id,
                    chunk_index=i,
                    chunk_data=b"\x00" * 1024,
                    timestamp_ms=i * 100.0,
                    duration_ms=100.0,
                )
            collector.end_session(session_id)

        threads = [
            threading.Thread(target=run_session, args=(f"session_{i}", f"voice_{i}"))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All sessions should be ended
        assert collector.get_active_session_count() == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests with storage and streaming."""

    def test_analytics_storage_integration(self, collector):
        """Test integration with analytics storage."""
        from axiom_vox.analytics.storage import AnalyticsStorage

        storage = AnalyticsStorage(":memory:")

        # Run a session
        collector.start_session("session_1", "voice_1")

        for i in range(10):
            collector.record_chunk(
                session_id="session_1",
                chunk_index=i,
                chunk_data=b"\x00" * 4096,
                timestamp_ms=i * 100.0,
                duration_ms=100.0,
            )

        final_metrics = collector.end_session("session_1")

        # Convert StreamSessionMetrics to StreamingSessionAnalytics for storage
        session_analytics = StreamingSessionAnalytics(
            session_id=final_metrics.session_id,
            voice_id=final_metrics.voice_id,
            timestamp=final_metrics.started_at,
            text_length=0,
            total_chunks=final_metrics.total_chunks,
            total_bytes=final_metrics.total_bytes,
            total_audio_duration_ms=final_metrics.total_audio_duration_ms,
            sentences_completed=final_metrics.sentences_completed,
            first_chunk_latency_ms=final_metrics.first_chunk_latency_ms,
            avg_chunk_latency_ms=final_metrics.avg_latency_ms,
            latency_std_ms=final_metrics.latency_std_ms,
            latency_min_ms=final_metrics.latency_min_ms if final_metrics.latency_min_ms != float("inf") else None,
            latency_max_ms=final_metrics.latency_max_ms,
            late_chunk_ratio=final_metrics.late_chunk_ratio,
            late_chunks=final_metrics.late_chunks,
            stutters_detected=final_metrics.stutters_detected,
            quality_drops=final_metrics.quality_drops,
            avg_stream_quality=final_metrics.get_current_quality_score(),
            streaming_rtf=final_metrics.real_time_factor,
        )

        # Save to storage
        storage.save_streaming_session(session_analytics)

        # Retrieve
        retrieved = storage.get_streaming_session("session_1")

        assert retrieved is not None
        assert retrieved["session_id"] == "session_1"
        assert retrieved["total_chunks"] == 10

    def test_event_storage_integration(self, collector_with_callback, events_captured):
        """Test event storage integration."""
        from axiom_vox.analytics.storage import AnalyticsStorage

        storage = AnalyticsStorage(":memory:")

        collector_with_callback.start_session("session_1", "voice_1")

        # First chunk - should trigger FIRST_CHUNK event
        collector_with_callback.record_chunk(
            session_id="session_1",
            chunk_index=0,
            chunk_data=b"\x00" * 4096,
            timestamp_ms=0.0,
            duration_ms=100.0,
        )

        # Trigger a stutter
        time.sleep(0.55)

        collector_with_callback.record_chunk(
            session_id="session_1",
            chunk_index=1,
            chunk_data=b"\x00" * 4096,
            timestamp_ms=100.0,
            duration_ms=100.0,
        )

        # Save events (convert to dict first)
        for event in events_captured:
            storage.save_streaming_event(event.to_dict())

        # Retrieve events
        events = storage.get_streaming_events(limit=10)

        assert len(events) > 0


# ============================================================================
# DEMO / SMOKE TEST
# ============================================================================

class TestSmoke:
    """Smoke tests to verify basic functionality."""

    def test_full_session_lifecycle(self):
        """Test complete session from start to end."""
        collector = StreamingAnalyticsCollector(
            late_threshold_ms=150.0,
            stutter_threshold_ms=400.0,
        )

        # Start
        metrics = collector.start_session("smoke_test", "voice_demo")
        assert metrics is not None

        # Record chunks
        for i in range(20):
            chunk = collector.record_chunk(
                session_id="smoke_test",
                chunk_index=i,
                chunk_data=b"\x00" * 2048,
                timestamp_ms=i * 80.0,
                duration_ms=80.0,
                is_sentence_end=(i % 10 == 9),
            )
            assert chunk is not None

        # Get metrics
        current = collector.get_current_metrics("smoke_test")
        assert current.total_chunks == 20
        assert current.total_bytes == 2048 * 20

        # Get recent chunks
        recent = collector.get_recent_chunks("smoke_test", count=5)
        assert len(recent) == 5

        # End
        final = collector.end_session("smoke_test")
        assert final.last_chunk_at is not None

        # Session gone
        assert collector.get_current_metrics("smoke_test") is None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
