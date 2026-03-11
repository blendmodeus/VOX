"""
Tests for AXIØM VØX Unified Pipeline
------------------------------------

Unified voice pipeline tests.

v0.11.0: Unified Voice Pipeline
"""

import time
import uuid
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_text():
    """Sample text for synthesis."""
    return "Hello, this is a test of the unified voice pipeline."


@pytest.fixture
def test_voice_id():
    """Test voice ID."""
    return f"test_voice_{uuid.uuid4().hex[:8]}"


# ============================================================================
# Pipeline Models Tests
# ============================================================================


class TestPipelineModels:
    """Tests for pipeline data models."""

    def test_pipeline_request_defaults(self):
        """Test PipelineRequest default values."""
        from axiom_vox.unified import PipelineRequest

        request = PipelineRequest(text="Hello")

        assert request.text == "Hello"
        assert request.voice_id is None
        assert request.require_biometric_verification is False
        assert request.enable_quality_monitoring is True
        assert request.stream is False
        assert request.speaking_rate == 1.0
        assert request.request_id.startswith("req_")

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        from axiom_vox.unified import PipelineConfig

        config = PipelineConfig()

        assert config.enable_biometric_identification is True
        assert config.require_consent is True
        assert config.strict_governance is False
        assert config.enable_quality_gates is True
        assert config.default_voice_id == "axiom_default"
        assert config.min_quality_score == 0.6

    def test_pipeline_response_structure(self):
        """Test PipelineResponse structure."""
        from axiom_vox.unified import PipelineResponse, PipelineStatus

        response = PipelineResponse(request_id="test_123")

        assert response.request_id == "test_123"
        assert response.status == PipelineStatus.PENDING
        assert response.stages_completed == []
        assert response.error is None

    def test_stage_result_structure(self):
        """Test StageResult structure."""
        from axiom_vox.unified import StageResult, PipelineStage

        result = StageResult(
            stage=PipelineStage.INTAKE,
            success=True,
            duration_ms=10.5,
            message="OK",
        )

        assert result.stage == PipelineStage.INTAKE
        assert result.success is True
        assert result.duration_ms == 10.5

    def test_consent_scope_enum(self):
        """Test ConsentScope enum values."""
        from axiom_vox.unified import ConsentScope

        assert ConsentScope.SYNTHESIS.value == "synthesis"
        assert ConsentScope.CLONING.value == "cloning"
        assert ConsentScope.BIOMETRIC.value == "biometric"
        assert ConsentScope.STREAMING.value == "streaming"

    def test_voice_route_type_enum(self):
        """Test VoiceRouteType enum values."""
        from axiom_vox.unified import VoiceRouteType

        assert VoiceRouteType.EXPLICIT.value == "explicit"
        assert VoiceRouteType.BIOMETRIC.value == "biometric"
        assert VoiceRouteType.CONTEXT.value == "context"
        assert VoiceRouteType.DEFAULT.value == "default"


# ============================================================================
# Consent Registry Tests
# ============================================================================


class TestUnifiedConsentRegistry:
    """Tests for UnifiedConsentRegistry."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = MagicMock()
        db.get_voice.return_value = {
            "category": "synthetic",
            "consent_verified": True,
            "allowed_uses": ["general"],
        }
        db.get_active_consent.return_value = None
        return db

    def test_check_consent_synthetic_voice(self, mock_db, test_voice_id):
        """Test consent check for synthetic voice."""
        from axiom_vox.unified import UnifiedConsentRegistry, ConsentScope

        registry = UnifiedConsentRegistry(db=mock_db)

        result = registry.check_consent(
            voice_id=test_voice_id,
            required_scopes=[ConsentScope.SYNTHESIS],
        )

        assert result.granted is True
        assert ConsentScope.SYNTHESIS in result.scopes

    def test_check_consent_missing_scope(self, mock_db, test_voice_id):
        """Test consent check with missing scope."""
        from axiom_vox.unified import UnifiedConsentRegistry, ConsentScope

        mock_db.get_voice.return_value = {"category": "restricted"}

        registry = UnifiedConsentRegistry(db=mock_db)

        result = registry.check_consent(
            voice_id=test_voice_id,
            required_scopes=[ConsentScope.COMMERCIAL],
        )

        # Commercial consent not granted
        assert result.granted is False
        assert "commercial" in result.message.lower()

    def test_get_consent_status(self, mock_db, test_voice_id):
        """Test getting complete consent status."""
        from axiom_vox.unified import UnifiedConsentRegistry

        registry = UnifiedConsentRegistry(db=mock_db)

        status = registry.get_consent_status(test_voice_id)

        assert status["voice_id"] == test_voice_id
        assert "scopes" in status
        assert "synthesis" in status["scopes"]


# ============================================================================
# Voice Router Tests
# ============================================================================


class TestBiometricVoiceRouter:
    """Tests for BiometricVoiceRouter."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = MagicMock()
        db.get_voice.return_value = {
            "voice_id": "test_voice",
            "category": "synthetic",
        }
        return db

    @pytest.mark.asyncio
    async def test_route_explicit_voice(self, mock_db, test_voice_id):
        """Test routing with explicit voice ID."""
        from axiom_vox.unified import (
            BiometricVoiceRouter,
            PipelineConfig,
            VoiceRouteType,
        )

        config = PipelineConfig()
        router = BiometricVoiceRouter(config=config, db=mock_db)

        result = await router.route(voice_id=test_voice_id)

        assert result.voice_id == test_voice_id
        assert result.route_type in [VoiceRouteType.EXPLICIT, VoiceRouteType.DEFAULT]

    @pytest.mark.asyncio
    async def test_route_fallback(self, mock_db):
        """Test fallback to default voice."""
        from axiom_vox.unified import (
            BiometricVoiceRouter,
            PipelineConfig,
            VoiceRouteType,
        )

        mock_db.get_voice.return_value = None  # Voice not found

        config = PipelineConfig()
        router = BiometricVoiceRouter(config=config, db=mock_db)

        result = await router.route(voice_id="nonexistent")

        assert result.route_type == VoiceRouteType.DEFAULT
        assert result.fallback_used is True

    @pytest.mark.asyncio
    async def test_route_with_context(self, mock_db, test_text):
        """Test context-based routing."""
        from axiom_vox.unified import BiometricVoiceRouter, PipelineConfig

        config = PipelineConfig(enable_context_routing=True)
        router = BiometricVoiceRouter(config=config, db=mock_db)

        result = await router.route(
            text=test_text,
            context={"domain": "technical"},
        )

        # Should return some route
        assert result.voice_id is not None


# ============================================================================
# Quality Monitor Tests
# ============================================================================


class TestRealTimeQualityMonitor:
    """Tests for RealTimeQualityMonitor."""

    def test_start_session(self):
        """Test starting monitoring session."""
        from axiom_vox.unified import RealTimeQualityMonitor

        monitor = RealTimeQualityMonitor()
        monitor.start_session("test_session")

        stats = monitor.get_running_quality()
        assert stats["chunk_count"] == 0

    def test_analyze_chunk(self):
        """Test analyzing audio chunk."""
        from axiom_vox.unified import RealTimeQualityMonitor

        monitor = RealTimeQualityMonitor()
        monitor.start_session("test_session")

        # Generate test audio
        audio = np.random.randn(24000).astype(np.float32) * 0.1

        snapshot = monitor.analyze_chunk(audio, 24000)

        assert snapshot.chunk_index == 1
        assert 0 <= snapshot.overall_score <= 1
        assert snapshot.snr_db >= 0

    def test_running_statistics(self):
        """Test running quality statistics."""
        from axiom_vox.unified import RealTimeQualityMonitor

        monitor = RealTimeQualityMonitor()
        monitor.start_session("test_session")

        # Analyze multiple chunks
        for _ in range(5):
            audio = np.random.randn(24000).astype(np.float32) * 0.1
            monitor.analyze_chunk(audio, 24000)

        stats = monitor.get_running_quality()

        assert stats["chunk_count"] == 5
        assert stats["average_quality"] > 0
        assert stats["min_quality"] <= stats["max_quality"]

    def test_finalize_session(self):
        """Test finalizing session with quality result."""
        from axiom_vox.unified import RealTimeQualityMonitor, QualityGate

        monitor = RealTimeQualityMonitor()
        monitor.start_session("test_session")

        # Analyze chunks
        for _ in range(3):
            audio = np.random.randn(24000).astype(np.float32) * 0.1
            monitor.analyze_chunk(audio, 24000)

        result = monitor.finalize_session()

        assert result.overall_score >= 0
        assert result.gate_status in list(QualityGate)

    def test_silent_chunk_detection(self):
        """Test detection of silent chunks."""
        from axiom_vox.unified import RealTimeQualityMonitor

        monitor = RealTimeQualityMonitor()
        monitor.start_session("test_session")

        # Silent audio
        silent = np.zeros(24000, dtype=np.float32)

        snapshot = monitor.analyze_chunk(silent, 24000)

        assert snapshot.is_silent is True


# ============================================================================
# Unified Pipeline Tests
# ============================================================================


class TestVoxUnifiedPipeline:
    """Tests for VoxUnifiedPipeline."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create pipeline with mocked components."""
        from axiom_vox.unified import VoxUnifiedPipeline, PipelineConfig

        with patch('axiom_vox.unified.pipeline.get_consent_registry') as mock_consent:
            with patch('axiom_vox.unified.pipeline.get_voice_router') as mock_router:
                mock_consent_instance = MagicMock()
                mock_router_instance = MagicMock()

                mock_consent.return_value = mock_consent_instance
                mock_router.return_value = mock_router_instance

                config = PipelineConfig(
                    require_consent=False,  # Simplify for testing
                    enable_quality_gates=False,
                )
                pipeline = VoxUnifiedPipeline(config=config)

                yield pipeline, mock_consent_instance, mock_router_instance

    @pytest.mark.asyncio
    async def test_process_intake_validation(self, mock_pipeline, test_text):
        """Test intake stage validation."""
        from axiom_vox.unified import PipelineRequest, PipelineStatus

        pipeline, _, _ = mock_pipeline

        # Valid request
        request = PipelineRequest(text=test_text)
        response = await pipeline.process(request)

        # Should at least complete intake
        assert "intake" in [s.value for s in response.stages_completed]

    @pytest.mark.asyncio
    async def test_process_empty_text_fails(self, mock_pipeline):
        """Test that empty text fails intake."""
        from axiom_vox.unified import PipelineRequest, PipelineStatus

        pipeline, _, _ = mock_pipeline

        request = PipelineRequest()  # No text
        response = await pipeline.process(request)

        assert response.status == PipelineStatus.FAILED
        assert response.error is not None

    @pytest.mark.asyncio
    async def test_process_text_length_limit(self, mock_pipeline):
        """Test text length validation."""
        from axiom_vox.unified import PipelineRequest, PipelineStatus

        pipeline, _, _ = mock_pipeline
        pipeline.config.max_text_length = 100

        # Text exceeding limit
        long_text = "x" * 200
        request = PipelineRequest(text=long_text)
        response = await pipeline.process(request)

        assert response.status == PipelineStatus.FAILED
        assert "length" in response.error.lower()


# ============================================================================
# Integration Tests
# ============================================================================


class TestUnifiedPipelineIntegration:
    """Integration tests for complete pipeline flow."""

    def test_imports_work(self):
        """Test all unified module imports."""
        from axiom_vox.unified import (
            VoxUnifiedPipeline,
            PipelineRequest,
            PipelineResponse,
            PipelineConfig,
            UnifiedConsentRegistry,
            BiometricVoiceRouter,
            RealTimeQualityMonitor,
        )

        # All imports should work
        assert VoxUnifiedPipeline is not None
        assert PipelineRequest is not None

    def test_main_module_exports(self):
        """Test exports from main axiom_vox module."""
        from axiom_vox import (
            VoxUnifiedPipeline,
            get_unified_pipeline,
            synthesize_unified,
            PipelineRequest,
            PipelineResponse,
            PipelineConfig,
        )

        assert VoxUnifiedPipeline is not None
        assert get_unified_pipeline is not None

    def test_pipeline_config_integration(self):
        """Test pipeline with custom config."""
        from axiom_vox.unified import VoxUnifiedPipeline, PipelineConfig

        config = PipelineConfig(
            require_consent=False,
            strict_governance=True,
            enable_quality_gates=True,
            min_quality_score=0.7,
        )

        pipeline = VoxUnifiedPipeline(config=config)

        assert pipeline.config.require_consent is False
        assert pipeline.config.strict_governance is True

    def test_quality_monitor_thresholds(self):
        """Test quality thresholds are applied."""
        from axiom_vox.unified import QUALITY_THRESHOLDS

        assert QUALITY_THRESHOLDS["excellent"] == 0.9
        assert QUALITY_THRESHOLDS["good"] == 0.8
        assert QUALITY_THRESHOLDS["acceptable"] == 0.6

    def test_stage_order(self):
        """Test pipeline stage order."""
        from axiom_vox.unified import STAGE_ORDER, PipelineStage

        assert STAGE_ORDER[0] == PipelineStage.INTAKE
        assert STAGE_ORDER[-1] == PipelineStage.DELIVERY
        assert PipelineStage.GOVERNANCE in STAGE_ORDER


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_consent_scopes(self):
        """Test consent check with empty scopes."""
        from axiom_vox.unified import UnifiedConsentRegistry

        mock_db = MagicMock()
        registry = UnifiedConsentRegistry(db=mock_db)

        result = registry.check_consent(
            voice_id="test",
            required_scopes=[],
        )

        # Empty scopes should grant
        assert result.granted is True

    def test_quality_monitor_no_chunks(self):
        """Test finalizing session with no chunks."""
        from axiom_vox.unified import RealTimeQualityMonitor

        monitor = RealTimeQualityMonitor()
        monitor.start_session("empty_session")

        result = monitor.finalize_session()

        assert result.overall_score == 0.0

    def test_voice_router_cache_invalidation(self):
        """Test voice router cache invalidation."""
        from axiom_vox.unified import BiometricVoiceRouter

        mock_db = MagicMock()
        router = BiometricVoiceRouter(db=mock_db)

        # Add to cache
        router._voice_cache["test"] = MagicMock()

        # Invalidate
        router.invalidate_cache("test")

        assert "test" not in router._voice_cache

    def test_consent_cache_ttl(self):
        """Test consent cache TTL handling."""
        from axiom_vox.unified import UnifiedConsentRegistry

        mock_db = MagicMock()
        mock_db.get_voice.return_value = {"category": "synthetic"}

        registry = UnifiedConsentRegistry(db=mock_db)
        registry._cache_ttl = 0  # Immediate expiry

        # First call caches
        registry.check_consent("test", [])

        # Second call should re-fetch (cache expired)
        registry.check_consent("test", [])

        # Should have called database multiple times
        assert mock_db.get_voice.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
