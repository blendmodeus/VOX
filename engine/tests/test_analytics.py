"""
Tests for AXIOM VOX Analytics
-----------------------------

Tests for:
- TechnicalQualityMetrics, SpectralQualityMetrics, NaturalnessMetrics
- VoxAudioAnalyzer: SNR, spectral analysis, naturalness estimation
- VoxMetricsCollector: Synthesis lifecycle tracking
- AnalyticsStorage: Database operations
- VoxAnalyticsService: Aggregation and trends
"""

import os
import tempfile
import time
import pytest
from typing import Optional


# ============================================================================
# MODELS TESTS
# ============================================================================

class TestTechnicalQualityMetrics:
    """Tests for TechnicalQualityMetrics."""

    def test_metrics_creation(self):
        """Test creating technical quality metrics."""
        from axiom_vox.analytics import TechnicalQualityMetrics

        metrics = TechnicalQualityMetrics(
            snr_db=28.5,
            peak_amplitude=0.85,
            rms_level_db=-18.0,
            dynamic_range_db=12.0,
            silence_ratio=0.05,
            clipping_samples=0,
            artifact_score=0.1,
        )

        assert metrics.snr_db == 28.5
        assert metrics.peak_amplitude == 0.85
        assert metrics.clipping_samples == 0

    def test_quality_score_computation(self):
        """Test quality score is between 0 and 1."""
        from axiom_vox.analytics import TechnicalQualityMetrics

        metrics = TechnicalQualityMetrics(
            snr_db=30.0,
            peak_amplitude=0.8,
            rms_level_db=-15.0,
            dynamic_range_db=10.0,
            silence_ratio=0.1,
            clipping_samples=0,
            artifact_score=0.1,
        )

        score = metrics.get_quality_score()
        assert 0.0 <= score <= 1.0

    def test_to_dict(self):
        """Test serialization to dict."""
        from axiom_vox.analytics import TechnicalQualityMetrics

        metrics = TechnicalQualityMetrics(
            snr_db=25.0,
            peak_amplitude=0.7,
            rms_level_db=-20.0,
            dynamic_range_db=8.0,
            silence_ratio=0.15,
            clipping_samples=5,
            artifact_score=0.2,
        )

        d = metrics.to_dict()
        assert d["snr_db"] == 25.0
        assert d["clipping_samples"] == 5


class TestSpectralQualityMetrics:
    """Tests for SpectralQualityMetrics."""

    def test_metrics_creation(self):
        """Test creating spectral quality metrics."""
        from axiom_vox.analytics import SpectralQualityMetrics

        metrics = SpectralQualityMetrics(
            spectral_centroid_hz=1500.0,
            spectral_bandwidth_hz=800.0,
            spectral_rolloff_hz=4000.0,
            spectral_flatness=0.15,
            spectral_contrast=0.7,
            harmonic_ratio=0.85,
        )

        assert metrics.spectral_centroid_hz == 1500.0
        assert metrics.spectral_flatness == 0.15

    def test_quality_score(self):
        """Test spectral quality score computation."""
        from axiom_vox.analytics import SpectralQualityMetrics

        metrics = SpectralQualityMetrics(
            spectral_centroid_hz=1500.0,
            spectral_bandwidth_hz=800.0,
            spectral_rolloff_hz=4000.0,
            spectral_flatness=0.15,
            spectral_contrast=0.7,
            harmonic_ratio=0.85,
        )

        score = metrics.get_quality_score()
        assert 0.0 <= score <= 1.0


class TestNaturalnessMetrics:
    """Tests for NaturalnessMetrics."""

    def test_from_mos(self):
        """Test creating naturalness from MOS estimate."""
        from axiom_vox.analytics import NaturalnessMetrics

        metrics = NaturalnessMetrics.from_mos(3.8)

        assert metrics.mos_estimate == 3.8
        assert 0.0 <= metrics.overall_naturalness <= 1.0
        assert 0.0 <= metrics.prosody_score <= 1.0


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""

    def test_compute_from_timing(self):
        """Test computing metrics from timing data."""
        from axiom_vox.analytics import PerformanceMetrics

        metrics = PerformanceMetrics.compute(
            synthesis_latency_ms=250.0,
            audio_duration_ms=2500.0,
            text_length=100,
        )

        assert metrics.synthesis_latency_ms == 250.0
        assert metrics.audio_duration_ms == 2500.0
        assert metrics.real_time_factor == 0.1  # 250/2500
        assert metrics.characters_per_second == 400.0  # 100/0.25


class TestSynthesisAnalytics:
    """Tests for SynthesisAnalytics."""

    def test_creation(self):
        """Test creating synthesis analytics."""
        from axiom_vox.analytics import (
            SynthesisAnalytics,
            TechnicalQualityMetrics,
            ComputationStatus,
        )

        analytics = SynthesisAnalytics(
            synthesis_id="synth_123",
            voice_id="test_voice",
            timestamp=time.time(),
            text_length=100,
            technical_quality=TechnicalQualityMetrics(
                snr_db=25.0,
                peak_amplitude=0.8,
                rms_level_db=-18.0,
                dynamic_range_db=12.0,
                silence_ratio=0.1,
                clipping_samples=0,
                artifact_score=0.1,
            ),
            computation_status=ComputationStatus.COMPLETE,
        )

        assert analytics.synthesis_id == "synth_123"
        assert analytics.voice_id == "test_voice"

    def test_quality_score(self):
        """Test composite quality score."""
        from axiom_vox.analytics import (
            SynthesisAnalytics,
            TechnicalQualityMetrics,
            NaturalnessMetrics,
        )

        analytics = SynthesisAnalytics(
            synthesis_id="synth_123",
            voice_id="test_voice",
            timestamp=time.time(),
            text_length=100,
            technical_quality=TechnicalQualityMetrics(
                snr_db=30.0,
                peak_amplitude=0.8,
                rms_level_db=-18.0,
                dynamic_range_db=12.0,
                silence_ratio=0.1,
                clipping_samples=0,
                artifact_score=0.1,
            ),
            naturalness=NaturalnessMetrics.from_mos(4.0),
        )

        score = analytics.get_quality_score()
        assert 0.0 <= score <= 1.0

    def test_quality_tier(self):
        """Test quality tier classification."""
        from axiom_vox.analytics import (
            SynthesisAnalytics,
            TechnicalQualityMetrics,
            QualityTier,
        )

        analytics = SynthesisAnalytics(
            synthesis_id="synth_123",
            voice_id="test_voice",
            timestamp=time.time(),
            text_length=100,
            technical_quality=TechnicalQualityMetrics(
                snr_db=35.0,
                peak_amplitude=0.8,
                rms_level_db=-15.0,
                dynamic_range_db=12.0,
                silence_ratio=0.05,
                clipping_samples=0,
                artifact_score=0.05,
            ),
        )

        tier = analytics.get_quality_tier()
        assert tier in [QualityTier.EXCELLENT, QualityTier.GOOD, QualityTier.ACCEPTABLE, QualityTier.POOR]

    def test_to_dict(self):
        """Test serialization."""
        from axiom_vox.analytics import SynthesisAnalytics

        analytics = SynthesisAnalytics(
            synthesis_id="synth_123",
            voice_id="test_voice",
            timestamp=time.time(),
            text_length=100,
        )

        d = analytics.to_dict()
        assert d["synthesis_id"] == "synth_123"
        assert "quality_score" in d


# ============================================================================
# ANALYZER TESTS
# ============================================================================

class TestVoxAudioAnalyzer:
    """Tests for VoxAudioAnalyzer."""

    def test_analyzer_creation(self):
        """Test creating analyzer."""
        from axiom_vox.analytics import VoxAudioAnalyzer

        analyzer = VoxAudioAnalyzer()
        assert analyzer is not None

    def test_compute_technical_metrics_without_numpy(self):
        """Test technical metrics fallback without numpy."""
        from axiom_vox.analytics import VoxAudioAnalyzer

        analyzer = VoxAudioAnalyzer()
        metrics = analyzer.compute_technical_metrics(None, 24000)

        # Should return default values
        assert metrics.snr_db == 20.0
        assert metrics.peak_amplitude == 0.5

    def test_compute_spectral_metrics_without_scipy(self):
        """Test spectral metrics fallback."""
        from axiom_vox.analytics import VoxAudioAnalyzer

        analyzer = VoxAudioAnalyzer()
        metrics = analyzer.compute_spectral_metrics(None, 24000)

        # Should return default values
        assert metrics.spectral_centroid_hz == 1500.0
        assert metrics.spectral_flatness == 0.2


# ============================================================================
# COLLECTOR TESTS
# ============================================================================

class TestVoxMetricsCollector:
    """Tests for VoxMetricsCollector."""

    def test_collector_creation(self):
        """Test creating collector."""
        from axiom_vox.analytics import VoxMetricsCollector

        collector = VoxMetricsCollector()
        assert collector is not None

    def test_synthesis_lifecycle(self):
        """Test start/end synthesis tracking."""
        from axiom_vox.analytics import VoxMetricsCollector

        collector = VoxMetricsCollector()

        # Start synthesis
        synthesis_id = collector.generate_synthesis_id()
        collector.start_synthesis(synthesis_id, "Hello world", "test_voice")

        # Simulate work
        time.sleep(0.01)

        # End synthesis
        perf = collector.end_synthesis(synthesis_id, audio_duration_ms=1000)

        assert perf.synthesis_latency_ms > 0
        assert perf.audio_duration_ms == 1000

    def test_get_analytics(self):
        """Test getting analytics for a synthesis."""
        from axiom_vox.analytics import VoxMetricsCollector

        collector = VoxMetricsCollector()

        synthesis_id = "test_synth_001"
        collector.start_synthesis(synthesis_id, "Test text", "voice_1")
        collector.end_synthesis(synthesis_id, audio_duration_ms=500)

        analytics = collector.get_analytics(synthesis_id)

        assert analytics is not None
        assert analytics.synthesis_id == synthesis_id
        assert analytics.voice_id == "voice_1"
        assert analytics.text_length == len("Test text")

    def test_record_governance(self):
        """Test recording governance decisions."""
        from axiom_vox.analytics import VoxMetricsCollector

        collector = VoxMetricsCollector()

        synthesis_id = "test_synth_002"
        collector.start_synthesis(synthesis_id, "Test", "voice_1")
        collector.record_governance(synthesis_id, "allow", content_emotion_match=0.9)
        collector.end_synthesis(synthesis_id)

        analytics = collector.get_analytics(synthesis_id)

        assert analytics.governance_action == "allow"
        assert analytics.content_emotion_match == 0.9

    def test_generate_synthesis_id(self):
        """Test generating unique IDs."""
        from axiom_vox.analytics import VoxMetricsCollector

        collector = VoxMetricsCollector()

        id1 = collector.generate_synthesis_id()
        id2 = collector.generate_synthesis_id()

        assert id1 != id2
        assert id1.startswith("synth_")


# ============================================================================
# STORAGE TESTS
# ============================================================================

class TestAnalyticsStorage:
    """Tests for AnalyticsStorage."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            yield db_path

    def test_storage_creation(self, temp_db):
        """Test creating storage."""
        from axiom_vox.analytics import AnalyticsStorage

        storage = AnalyticsStorage(db_path=temp_db)
        assert storage is not None
        storage.close()

    def test_save_and_retrieve(self, temp_db):
        """Test saving and retrieving analytics."""
        from axiom_vox.analytics import (
            AnalyticsStorage,
            SynthesisAnalytics,
            TechnicalQualityMetrics,
            PerformanceMetrics,
            ComputationStatus,
        )

        storage = AnalyticsStorage(db_path=temp_db)

        analytics = SynthesisAnalytics(
            synthesis_id="synth_001",
            voice_id="test_voice",
            timestamp=time.time(),
            text_length=100,
            technical_quality=TechnicalQualityMetrics(
                snr_db=25.0,
                peak_amplitude=0.8,
                rms_level_db=-18.0,
                dynamic_range_db=12.0,
                silence_ratio=0.1,
                clipping_samples=0,
                artifact_score=0.1,
            ),
            performance=PerformanceMetrics(
                synthesis_latency_ms=200,
                audio_duration_ms=2000,
                real_time_factor=0.1,
                characters_per_second=500,
            ),
            computation_status=ComputationStatus.COMPLETE,
        )

        storage.save_synthesis_metrics(analytics)
        retrieved = storage.get_synthesis_metrics("synth_001")

        assert retrieved is not None
        assert retrieved.synthesis_id == "synth_001"

        storage.close()

    def test_system_summary(self, temp_db):
        """Test getting system summary."""
        from axiom_vox.analytics import (
            AnalyticsStorage,
            SynthesisAnalytics,
            PerformanceMetrics,
            ComputationStatus,
        )

        storage = AnalyticsStorage(db_path=temp_db)

        # Add some test data
        for i in range(5):
            analytics = SynthesisAnalytics(
                synthesis_id=f"synth_{i:03d}",
                voice_id="test_voice",
                timestamp=time.time(),
                text_length=100,
                performance=PerformanceMetrics(
                    synthesis_latency_ms=200 + i * 50,
                    audio_duration_ms=2000,
                    real_time_factor=0.1,
                    characters_per_second=500,
                ),
                computation_status=ComputationStatus.COMPLETE,
            )
            storage.save_synthesis_metrics(analytics)

        summary = storage.get_system_summary(period_days=1)

        assert summary.total_syntheses == 5
        assert summary.unique_voices == 1

        storage.close()


# ============================================================================
# SERVICE TESTS
# ============================================================================

class TestVoxAnalyticsService:
    """Tests for VoxAnalyticsService."""

    @pytest.fixture
    def service(self):
        """Create service with temp storage."""
        from axiom_vox.analytics import AnalyticsStorage, VoxAnalyticsService

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = AnalyticsStorage(db_path=db_path)
            service = VoxAnalyticsService(storage, enable_event_detection=True)
            yield service
            service.stop_workers()
            storage.close()

    def test_service_creation(self, service):
        """Test creating service."""
        assert service is not None

    def test_start_stop_workers(self, service):
        """Test worker lifecycle."""
        service.start_workers()
        assert len(service._workers) == service.num_workers

        service.stop_workers()
        assert len(service._workers) == 0

    def test_record_and_summarize(self, service):
        """Test recording and summarizing."""
        from axiom_vox.analytics import (
            SynthesisAnalytics,
            PerformanceMetrics,
            ComputationStatus,
        )

        # Record some analytics (using sync method)
        for i in range(3):
            analytics = SynthesisAnalytics(
                synthesis_id=f"synth_{i:03d}",
                voice_id="test_voice",
                timestamp=time.time(),
                text_length=100,
                performance=PerformanceMetrics(
                    synthesis_latency_ms=200,
                    audio_duration_ms=2000,
                    real_time_factor=0.1,
                    characters_per_second=500,
                ),
                computation_status=ComputationStatus.COMPLETE,
            )
            service.record_synthesis_sync(analytics)

        # Get summary (sync via storage directly)
        summary = service.storage.get_system_summary(period_days=1)
        assert summary.total_syntheses == 3


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAnalyticsIntegration:
    """Integration tests for analytics workflow."""

    def test_full_workflow(self):
        """Test complete analytics workflow."""
        from axiom_vox.analytics import (
            VoxMetricsCollector,
            AnalyticsStorage,
            VoxAnalyticsService,
            ComputationStatus,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Create components
            collector = VoxMetricsCollector()
            storage = AnalyticsStorage(db_path=db_path)
            service = VoxAnalyticsService(storage, enable_event_detection=False)

            # Simulate synthesis
            synthesis_id = collector.generate_synthesis_id()
            collector.start_synthesis(synthesis_id, "Hello world", "test_voice")
            time.sleep(0.01)
            collector.end_synthesis(synthesis_id, audio_duration_ms=1000)
            collector.record_governance(synthesis_id, "allow")

            # Get analytics
            analytics = collector.get_analytics(synthesis_id)
            assert analytics is not None

            # Save to storage
            service.record_synthesis_sync(analytics)

            # Retrieve
            retrieved = storage.get_synthesis_metrics(synthesis_id)
            assert retrieved is not None

            # Cleanup
            storage.close()


# ============================================================================
# MODULE EXPORTS TEST
# ============================================================================

class TestAnalyticsExports:
    """Test module exports."""

    def test_all_exports_available(self):
        """Test all expected classes are exported."""
        from axiom_vox.analytics import (
            # Enums
            ComputationStatus,
            QualityTier,
            AnalyticsEventType,
            EventSeverity,
            # Metrics
            TechnicalQualityMetrics,
            SpectralQualityMetrics,
            NaturalnessMetrics,
            PerformanceMetrics,
            # Containers
            SynthesisAnalytics,
            VoiceAggregateMetrics,
            SystemAnalyticsSummary,
            # Components
            VoxAudioAnalyzer,
            VoxMetricsCollector,
            AnalyticsStorage,
            VoxAnalyticsService,
        )

        assert ComputationStatus is not None
        assert TechnicalQualityMetrics is not None
        assert SynthesisAnalytics is not None
        assert VoxAudioAnalyzer is not None
        assert VoxMetricsCollector is not None
        assert AnalyticsStorage is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
