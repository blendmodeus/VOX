"""
Tests for VØX Streaming TTS Infrastructure
------------------------------------------

Tests for:
- SentenceSegmenter: Text segmentation
- StreamConfig: Configuration
- AudioChunk/StreamMessage: Data structures
- StreamSession: Session management
- StreamingSynthesizer: Streaming synthesis
- StreamManager: Full integration
"""

import asyncio
import os
import pytest
import time
from typing import List


# ============================================================================
# SENTENCE SEGMENTATION TESTS
# ============================================================================

class TestSentenceSegmenter:
    """Tests for sentence segmentation."""

    def test_basic_segmentation(self):
        """Test basic sentence splitting."""
        from axiom_vox.streaming import SentenceSegmenter

        text = "Hello world. How are you?"
        sentences = SentenceSegmenter.segment(text)

        assert len(sentences) == 2
        assert sentences[0] == "Hello world."
        assert sentences[1] == "How are you?"

    def test_multiple_punctuation(self):
        """Test different sentence endings."""
        from axiom_vox.streaming import SentenceSegmenter

        text = "Is this a question? Yes it is! That is great."
        sentences = SentenceSegmenter.segment(text)

        assert len(sentences) == 3
        assert sentences[0] == "Is this a question?"
        assert sentences[1] == "Yes it is!"
        assert sentences[2] == "That is great."

    def test_abbreviations(self):
        """Test that abbreviations don't split sentences."""
        from axiom_vox.streaming import SentenceSegmenter

        text = "Dr. Smith said hello. Mrs. Jones agreed."
        sentences = SentenceSegmenter.segment(text)

        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]
        assert "Mrs. Jones" in sentences[1]

    def test_empty_text(self):
        """Test empty input."""
        from axiom_vox.streaming import SentenceSegmenter

        assert SentenceSegmenter.segment("") == []
        assert SentenceSegmenter.segment("   ") == []
        assert SentenceSegmenter.segment(None) == []

    def test_no_punctuation(self):
        """Test text without sentence-ending punctuation."""
        from axiom_vox.streaming import SentenceSegmenter

        text = "This text has no ending punctuation"
        sentences = SentenceSegmenter.segment(text)

        assert len(sentences) == 1
        assert sentences[0] == text

    def test_count_sentences(self):
        """Test sentence counting."""
        from axiom_vox.streaming import SentenceSegmenter

        text = "One. Two. Three."
        count = SentenceSegmenter.count_sentences(text)
        assert count == 3


# ============================================================================
# STREAM CONFIG TESTS
# ============================================================================

class TestStreamConfig:
    """Tests for StreamConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from axiom_vox.streaming import StreamConfig

        config = StreamConfig()

        assert config.chunk_size_bytes == 4096
        assert config.sample_rate == 24000
        assert config.max_first_chunk_ms == 500
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        from axiom_vox.streaming import StreamConfig

        config = StreamConfig(
            chunk_size_bytes=8192,
            sample_rate=48000,
            max_first_chunk_ms=300,
        )

        assert config.chunk_size_bytes == 8192
        assert config.sample_rate == 48000
        assert config.max_first_chunk_ms == 300


# ============================================================================
# AUDIO CHUNK TESTS
# ============================================================================

class TestAudioChunk:
    """Tests for AudioChunk."""

    def test_chunk_creation(self):
        """Test creating an audio chunk."""
        from axiom_vox.streaming import AudioChunk

        chunk = AudioChunk(
            data=b'\x00' * 4096,
            index=0,
            timestamp_ms=100.0,
            duration_ms=170.0,
            is_sentence_end=True,
            sentence_index=0,
        )

        assert len(chunk.data) == 4096
        assert chunk.index == 0
        assert chunk.is_sentence_end == True
        assert chunk.sentence_index == 0

    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        from axiom_vox.streaming import AudioChunk

        chunk = AudioChunk(
            data=b'\x00' * 1024,
            index=5,
            timestamp_ms=500.0,
            duration_ms=42.5,
        )

        d = chunk.to_dict()

        assert d["index"] == 5
        assert d["timestamp_ms"] == 500.0
        assert d["duration_ms"] == 42.5
        assert d["size_bytes"] == 1024
        assert "data" not in d  # Binary data not included


# ============================================================================
# STREAM MESSAGE TESTS
# ============================================================================

class TestStreamMessage:
    """Tests for StreamMessage."""

    def test_started_message(self):
        """Test started message creation."""
        from axiom_vox.streaming import StreamMessage, MessageType

        msg = StreamMessage(
            type=MessageType.STARTED,
            request_id="test_123",
            metadata={"total_sentences": 3},
        )

        assert msg.type == MessageType.STARTED
        assert msg.request_id == "test_123"
        assert msg.metadata["total_sentences"] == 3

    def test_message_to_json(self):
        """Test message JSON serialization."""
        from axiom_vox.streaming import StreamMessage, MessageType

        msg = StreamMessage(
            type=MessageType.PROGRESS,
            request_id="test_456",
            progress={"chunks_sent": 10},
        )

        j = msg.to_json()

        assert j["type"] == "progress"
        assert j["request_id"] == "test_456"
        assert j["progress"]["chunks_sent"] == 10
        assert "timestamp" in j

    def test_error_message(self):
        """Test error message."""
        from axiom_vox.streaming import StreamMessage, MessageType

        msg = StreamMessage(
            type=MessageType.ERROR,
            request_id="test_error",
            error="Something went wrong",
        )

        j = msg.to_json()
        assert j["type"] == "error"
        assert j["error"] == "Something went wrong"


# ============================================================================
# STREAM SESSION TESTS
# ============================================================================

class TestStreamSession:
    """Tests for StreamSession."""

    def test_session_creation(self):
        """Test creating a session."""
        from axiom_vox.streaming import StreamSession

        session = StreamSession(
            request_id="session_abc",
            text="Hello world.",
            voice_id="default",
        )

        assert session.request_id == "session_abc"
        assert session.text == "Hello world."
        assert session.state == StreamSession.State.PENDING

    def test_session_progress(self):
        """Test session progress tracking."""
        from axiom_vox.streaming import StreamSession

        session = StreamSession(
            request_id="session_xyz",
            text="Test.",
            voice_id="default",
        )

        session.total_sentences = 3
        session.sentences_completed = 1
        session.chunks_sent = 5
        session.bytes_sent = 20480
        session.started_at = time.time()
        session.first_chunk_at = time.time() + 0.3

        progress = session.get_progress()

        assert progress["total_sentences"] == 3
        assert progress["sentences_completed"] == 1
        assert progress["chunks_sent"] == 5
        assert progress["bytes_sent"] == 20480
        assert progress["latency_to_first_chunk_ms"] is not None

    def test_session_states(self):
        """Test session state transitions."""
        from axiom_vox.streaming import StreamSession

        session = StreamSession(
            request_id="test",
            text="Test.",
            voice_id="default",
        )

        # Test all state values exist
        assert StreamSession.State.PENDING.value == "pending"
        assert StreamSession.State.GOVERNING.value == "governing"
        assert StreamSession.State.STREAMING.value == "streaming"
        assert StreamSession.State.COMPLETED.value == "completed"
        assert StreamSession.State.ERROR.value == "error"
        assert StreamSession.State.CANCELLED.value == "cancelled"
        assert StreamSession.State.PAUSED.value == "paused"


# ============================================================================
# MESSAGE TYPE TESTS
# ============================================================================

class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_types_exist(self):
        """Test all message types exist."""
        from axiom_vox.streaming import MessageType

        # Lifecycle
        assert MessageType.STARTED.value == "started"
        assert MessageType.PROGRESS.value == "progress"
        assert MessageType.COMPLETED.value == "completed"
        assert MessageType.ERROR.value == "error"
        assert MessageType.CANCELLED.value == "cancelled"

        # Audio
        assert MessageType.CHUNK.value == "chunk"
        assert MessageType.SENTENCE_BOUNDARY.value == "sentence_boundary"

        # Governance
        assert MessageType.GOVERNANCE_PASSED.value == "governance_passed"
        assert MessageType.GOVERNANCE_REPAIRED.value == "governance_repaired"
        assert MessageType.GOVERNANCE_REFUSED.value == "governance_refused"

        # Control
        assert MessageType.PAUSE.value == "pause"
        assert MessageType.RESUME.value == "resume"
        assert MessageType.CANCEL.value == "cancel"


# ============================================================================
# STREAMING SYNTHESIZER TESTS
# ============================================================================

class TestStreamingSynthesizer:
    """Tests for StreamingSynthesizer."""

    def test_placeholder_streaming(self):
        """Test placeholder streaming generates audio."""
        from axiom_vox.streaming import StreamingSynthesizer, StreamConfig
        from axiom_vox.synthesis import VoxSynthesizer

        async def _test():
            synth = VoxSynthesizer()
            streaming = StreamingSynthesizer(synth, StreamConfig())

            text = "Hello world. This is a test."
            chunks = []

            async for chunk in streaming.synthesize_stream_placeholder(text, "default"):
                chunks.append(chunk)

            assert len(chunks) > 0
            # Should have chunks from 2 sentences
            sentence_ends = [c for c in chunks if c.is_sentence_end]
            assert len(sentence_ends) == 2

        asyncio.run(_test())

    def test_placeholder_timing(self):
        """Test placeholder streaming has correct timing metadata."""
        from axiom_vox.streaming import StreamingSynthesizer, StreamConfig
        from axiom_vox.synthesis import VoxSynthesizer

        async def _test():
            synth = VoxSynthesizer()
            streaming = StreamingSynthesizer(synth, StreamConfig())

            text = "Test sentence."
            chunks = []

            async for chunk in streaming.synthesize_stream_placeholder(text, "default"):
                chunks.append(chunk)

            # First chunk should have timestamp
            assert chunks[0].timestamp_ms >= 0

            # Timestamps should increase
            for i in range(1, len(chunks)):
                assert chunks[i].timestamp_ms >= chunks[i-1].timestamp_ms

        asyncio.run(_test())


# ============================================================================
# STREAM MANAGER TESTS
# ============================================================================

class TestStreamManager:
    """Tests for StreamManager."""

    def test_create_session(self):
        """Test session creation."""
        from axiom_vox.streaming import StreamManager, StreamConfig
        from axiom_vox.synthesis import get_synthesizer
        from axiom_vox.vox_governor import get_governor

        manager = StreamManager(
            synthesizer=get_synthesizer(),
            governor=get_governor(),
            config=StreamConfig(),
        )

        session = manager.create_session(
            text="Hello world.",
            voice_id="default",
        )

        assert session.request_id.startswith("stream_")
        assert session.text == "Hello world."
        assert session.voice_id == "default"

    def test_get_session(self):
        """Test retrieving a session."""
        from axiom_vox.streaming import StreamManager, StreamConfig
        from axiom_vox.synthesis import get_synthesizer
        from axiom_vox.vox_governor import get_governor

        manager = StreamManager(
            synthesizer=get_synthesizer(),
            governor=get_governor(),
        )

        session = manager.create_session("Test.", "default")
        retrieved = manager.get_session(session.request_id)

        assert retrieved is not None
        assert retrieved.request_id == session.request_id

    def test_cancel_session(self):
        """Test cancelling a session."""
        from axiom_vox.streaming import StreamManager, StreamSession
        from axiom_vox.synthesis import get_synthesizer
        from axiom_vox.vox_governor import get_governor

        manager = StreamManager(
            synthesizer=get_synthesizer(),
            governor=get_governor(),
        )

        session = manager.create_session("Test.", "default")
        cancelled = manager.cancel_session(session.request_id)

        assert cancelled == True
        assert session.state == StreamSession.State.CANCELLED

    def test_remove_session(self):
        """Test removing a session."""
        from axiom_vox.streaming import StreamManager
        from axiom_vox.synthesis import get_synthesizer
        from axiom_vox.vox_governor import get_governor

        manager = StreamManager(
            synthesizer=get_synthesizer(),
            governor=get_governor(),
        )

        session = manager.create_session("Test.", "default")
        request_id = session.request_id

        manager.remove_session(request_id)

        assert manager.get_session(request_id) is None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestStreamingIntegration:
    """Integration tests for full streaming pipeline."""

    def test_full_streaming_flow(self):
        """Test complete streaming flow."""
        from axiom_vox.streaming import (
            StreamManager, StreamSession, MessageType
        )
        from axiom_vox.synthesis import get_synthesizer
        from axiom_vox.vox_governor import get_governor

        async def _test():
            manager = StreamManager(
                synthesizer=get_synthesizer(),
                governor=get_governor(),
            )

            # Create session
            session = manager.create_session(
                text="Hello world. This is a test.",
                voice_id="default",
            )

            # Run governance
            passed = await manager.run_governance(session)
            assert passed == True
            assert session.total_sentences == 2

            # Stream
            messages = []
            async for msg in manager.stream(session):
                messages.append(msg)

            # Verify message sequence
            msg_types = [m.type for m in messages]

            assert MessageType.STARTED in msg_types
            assert MessageType.CHUNK in msg_types
            assert MessageType.COMPLETED in msg_types

            # Verify chunks received
            chunks = [m for m in messages if m.type == MessageType.CHUNK]
            assert len(chunks) > 0

            # Verify sentence boundaries
            boundaries = [m for m in messages if m.type == MessageType.SENTENCE_BOUNDARY]
            assert len(boundaries) == 2

        asyncio.run(_test())

    def test_streaming_with_governance_report(self):
        """Test that governance report is included."""
        from axiom_vox.streaming import StreamManager, MessageType
        from axiom_vox.synthesis import get_synthesizer
        from axiom_vox.vox_governor import get_governor

        async def _test():
            manager = StreamManager(
                synthesizer=get_synthesizer(),
                governor=get_governor(),
            )

            session = manager.create_session("Test sentence.", "default")
            await manager.run_governance(session)

            messages = []
            async for msg in manager.stream(session):
                messages.append(msg)

            # Find governance message
            governance_msgs = [
                m for m in messages
                if m.type in (
                    MessageType.GOVERNANCE_PASSED,
                    MessageType.GOVERNANCE_REPAIRED,
                )
            ]

            assert len(governance_msgs) >= 1

        asyncio.run(_test())


# ============================================================================
# MODULE EXPORTS TEST
# ============================================================================

class TestModuleExports:
    """Tests for module exports."""

    def test_streaming_exports(self):
        """Test that all public classes are exported."""
        from axiom_vox.streaming import (
            StreamConfig,
            MessageType,
            AudioChunk,
            StreamMessage,
            StreamSession,
            SentenceSegmenter,
            StreamingSynthesizer,
            StreamManager,
            get_stream_manager,
            reset_stream_manager,
        )

        # Just verify they're importable
        assert StreamConfig is not None
        assert MessageType is not None
        assert AudioChunk is not None
        assert StreamManager is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
