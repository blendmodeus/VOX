"""
AXIOM VOX Analytics Models
--------------------------

Dataclasses for voice analytics metrics.

Categories:
- TechnicalQualityMetrics: Basic audio quality (SNR, clipping, etc.)
- SpectralQualityMetrics: Spectral analysis (centroid, flatness, etc.)
- NaturalnessMetrics: Speech naturalness estimation (MOS, prosody)
- PerformanceMetrics: Synthesis performance (latency, throughput)
- VoiceConsistencyMetrics: Voice identity consistency
- SynthesisAnalytics: Complete analytics for a single synthesis
- VoiceAggregateMetrics: Aggregated metrics over time

Usage:
    from axiom_vox.analytics import SynthesisAnalytics, TechnicalQualityMetrics
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import json


class ComputationStatus(str, Enum):
    """Status of analytics computation."""
    PENDING = "pending"
    COMPUTING = "computing"
    COMPLETE = "complete"
    FAILED = "failed"


class QualityTier(str, Enum):
    """Quality classification tiers."""
    EXCELLENT = "excellent"  # > 0.85
    GOOD = "good"            # 0.7 - 0.85
    ACCEPTABLE = "acceptable"  # 0.5 - 0.7
    POOR = "poor"            # < 0.5


# ============================================================================
# TECHNICAL QUALITY METRICS
# ============================================================================

@dataclass
class TechnicalQualityMetrics:
    """
    Technical audio quality metrics.

    Measures basic signal quality characteristics that can be
    computed quickly from the audio waveform.
    """
    snr_db: float                    # Signal-to-noise ratio (dB)
    peak_amplitude: float            # Maximum amplitude (0-1)
    rms_level_db: float              # Root mean square level (dB)
    dynamic_range_db: float          # Peak to RMS ratio (dB)
    silence_ratio: float             # Fraction of audio below threshold
    clipping_samples: int            # Number of clipped samples
    artifact_score: float            # Artifact detection score (0-1, lower is better)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_quality_score(self) -> float:
        """
        Compute composite quality score (0-1).

        Weights:
        - SNR: 30% (higher is better, normalized to 0-40dB range)
        - Artifact: 30% (lower is better, inverted)
        - Clipping: 20% (fewer is better)
        - Dynamic range: 20% (6-20dB is ideal)
        """
        # SNR score (0-40dB normalized)
        snr_score = min(1.0, max(0.0, self.snr_db / 40.0))

        # Artifact score (inverted, lower artifacts = higher score)
        artifact_quality = 1.0 - self.artifact_score

        # Clipping score (binary: any clipping is bad)
        clipping_score = 1.0 if self.clipping_samples == 0 else max(0.0, 1.0 - self.clipping_samples / 100)

        # Dynamic range score (6-20dB is ideal)
        if 6 <= self.dynamic_range_db <= 20:
            dr_score = 1.0
        elif self.dynamic_range_db < 6:
            dr_score = self.dynamic_range_db / 6.0
        else:
            dr_score = max(0.5, 1.0 - (self.dynamic_range_db - 20) / 20)

        return 0.3 * snr_score + 0.3 * artifact_quality + 0.2 * clipping_score + 0.2 * dr_score


# ============================================================================
# SPECTRAL QUALITY METRICS
# ============================================================================

@dataclass
class SpectralQualityMetrics:
    """
    Spectral analysis metrics.

    Measures frequency-domain characteristics that indicate
    voice quality and tonal properties.
    """
    spectral_centroid_hz: float      # Brightness measure (weighted mean freq)
    spectral_bandwidth_hz: float     # Spread of frequencies
    spectral_rolloff_hz: float       # Frequency below which 85% of energy
    spectral_flatness: float         # Tonality measure (0=tonal, 1=noisy)
    spectral_contrast: float         # Peak-to-valley ratio in spectrum
    harmonic_ratio: float            # Harmonic to noise ratio

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_quality_score(self) -> float:
        """
        Compute spectral quality score (0-1).

        Good speech typically has:
        - Spectral centroid: 500-3000 Hz
        - Low spectral flatness (more tonal)
        - High harmonic ratio
        """
        # Centroid score (500-3000 Hz is ideal for speech)
        if 500 <= self.spectral_centroid_hz <= 3000:
            centroid_score = 1.0
        elif self.spectral_centroid_hz < 500:
            centroid_score = self.spectral_centroid_hz / 500
        else:
            centroid_score = max(0.5, 1.0 - (self.spectral_centroid_hz - 3000) / 5000)

        # Flatness score (lower is better for speech)
        flatness_score = 1.0 - min(1.0, self.spectral_flatness)

        # Harmonic ratio score (higher is better)
        harmonic_score = min(1.0, self.harmonic_ratio)

        return 0.3 * centroid_score + 0.4 * flatness_score + 0.3 * harmonic_score


# ============================================================================
# NATURALNESS METRICS
# ============================================================================

@dataclass
class NaturalnessMetrics:
    """
    Speech naturalness estimation metrics.

    Estimates how human-like the synthesized speech sounds.
    Heavy computation - typically computed async.
    """
    mos_estimate: float              # Mean Opinion Score estimate (1-5)
    prosody_score: float             # Rhythm/intonation naturalness (0-1)
    articulation_score: float        # Consonant/vowel clarity (0-1)
    pacing_variance: float           # Speaking rate consistency (lower is more natural)
    breath_naturalness: float        # Natural pause patterns (0-1)
    overall_naturalness: float       # Composite score (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_mos(cls, mos: float) -> "NaturalnessMetrics":
        """Create NaturalnessMetrics from MOS estimate with derived values."""
        # Derive other scores from MOS (simplified)
        normalized = (mos - 1) / 4  # Normalize 1-5 to 0-1
        return cls(
            mos_estimate=mos,
            prosody_score=normalized * 0.95,
            articulation_score=normalized * 0.9,
            pacing_variance=max(0.05, 0.3 - normalized * 0.25),
            breath_naturalness=normalized * 0.85,
            overall_naturalness=normalized,
        )


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

@dataclass
class PerformanceMetrics:
    """
    Synthesis performance metrics.

    Measures latency, throughput, and resource usage.
    """
    synthesis_latency_ms: float      # Time to generate audio
    first_chunk_latency_ms: Optional[float] = None  # Time to first byte (streaming)
    audio_duration_ms: float = 0.0   # Length of generated audio
    real_time_factor: float = 0.0    # synthesis_time / audio_duration
    characters_per_second: float = 0.0  # Text throughput
    tokens_per_second: float = 0.0   # Model throughput (if available)
    memory_peak_mb: Optional[float] = None  # Peak memory usage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def compute(
        cls,
        synthesis_latency_ms: float,
        audio_duration_ms: float,
        text_length: int,
        first_chunk_latency_ms: Optional[float] = None,
    ) -> "PerformanceMetrics":
        """Compute performance metrics from timing data."""
        rtf = synthesis_latency_ms / audio_duration_ms if audio_duration_ms > 0 else 0
        cps = (text_length / synthesis_latency_ms) * 1000 if synthesis_latency_ms > 0 else 0

        return cls(
            synthesis_latency_ms=synthesis_latency_ms,
            first_chunk_latency_ms=first_chunk_latency_ms,
            audio_duration_ms=audio_duration_ms,
            real_time_factor=rtf,
            characters_per_second=cps,
        )


# ============================================================================
# VOICE CONSISTENCY METRICS
# ============================================================================

@dataclass
class VoiceConsistencyMetrics:
    """
    Voice identity and consistency metrics.

    Measures how consistent the voice sounds across utterances
    and how well it matches the reference voice.
    """
    reference_similarity: float      # Cosine similarity to reference (0-1)
    within_session_variance: float   # Embedding variance across utterances
    emotional_accuracy: float        # Match between intended and detected emotion
    pitch_consistency: float         # F0 stability across utterances (0-1)
    timbre_consistency: float        # Spectral envelope stability (0-1)
    speaker_embedding: Optional[List[float]] = None  # 256-dim embedding

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes large embedding)."""
        d = asdict(self)
        d.pop('speaker_embedding', None)  # Don't include large embedding in dict
        return d

    def get_consistency_score(self) -> float:
        """Compute composite consistency score (0-1)."""
        return (
            0.4 * self.reference_similarity +
            0.2 * (1.0 - min(1.0, self.within_session_variance)) +
            0.2 * self.pitch_consistency +
            0.2 * self.timbre_consistency
        )


# ============================================================================
# SYNTHESIS ANALYTICS (COMPLETE)
# ============================================================================

@dataclass
class SynthesisAnalytics:
    """
    Complete analytics for a single synthesis operation.

    Combines all metric categories with metadata.
    """
    synthesis_id: str
    voice_id: str
    timestamp: float
    text_length: int

    # Quality metrics (may be computed async)
    technical_quality: Optional[TechnicalQualityMetrics] = None
    spectral_quality: Optional[SpectralQualityMetrics] = None
    naturalness: Optional[NaturalnessMetrics] = None
    voice_consistency: Optional[VoiceConsistencyMetrics] = None
    performance: Optional[PerformanceMetrics] = None

    # Governance integration
    governance_action: Optional[str] = None
    content_emotion_match: Optional[float] = None
    manipulation_detected: bool = False

    # Async computation status
    computation_status: ComputationStatus = ComputationStatus.PENDING
    computation_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "synthesis_id": self.synthesis_id,
            "voice_id": self.voice_id,
            "timestamp": self.timestamp,
            "text_length": self.text_length,
            "technical_quality": self.technical_quality.to_dict() if self.technical_quality else None,
            "spectral_quality": self.spectral_quality.to_dict() if self.spectral_quality else None,
            "naturalness": self.naturalness.to_dict() if self.naturalness else None,
            "voice_consistency": self.voice_consistency.to_dict() if self.voice_consistency else None,
            "performance": self.performance.to_dict() if self.performance else None,
            "governance_action": self.governance_action,
            "content_emotion_match": self.content_emotion_match,
            "manipulation_detected": self.manipulation_detected,
            "computation_status": self.computation_status.value,
            "computation_error": self.computation_error,
            "quality_score": self.get_quality_score(),
        }

    def get_quality_score(self) -> float:
        """
        Compute composite quality score (0-1).

        Weighted combination of available metrics.
        """
        scores = []
        weights = []

        if self.technical_quality:
            scores.append(self.technical_quality.get_quality_score())
            weights.append(0.25)

        if self.spectral_quality:
            scores.append(self.spectral_quality.get_quality_score())
            weights.append(0.25)

        if self.naturalness:
            scores.append(self.naturalness.overall_naturalness)
            weights.append(0.35)

        if self.voice_consistency:
            scores.append(self.voice_consistency.get_consistency_score())
            weights.append(0.15)

        if not scores:
            return 0.0

        # Normalize weights
        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    def get_quality_tier(self) -> QualityTier:
        """Classify quality into tier."""
        score = self.get_quality_score()
        if score >= 0.85:
            return QualityTier.EXCELLENT
        elif score >= 0.7:
            return QualityTier.GOOD
        elif score >= 0.5:
            return QualityTier.ACCEPTABLE
        else:
            return QualityTier.POOR

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> "SynthesisAnalytics":
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(
            synthesis_id=d["synthesis_id"],
            voice_id=d["voice_id"],
            timestamp=d["timestamp"],
            text_length=d["text_length"],
            computation_status=ComputationStatus(d.get("computation_status", "pending")),
        )


# ============================================================================
# STREAMING SESSION ANALYTICS
# ============================================================================

@dataclass
class StreamingSessionAnalytics:
    """
    Complete analytics for a streaming synthesis session.

    Extends SynthesisAnalytics with streaming-specific metrics
    like per-chunk latency, stutters, and real-time monitoring.
    """
    # Base identification
    session_id: str
    voice_id: str
    timestamp: float
    text_length: int

    # Streaming-specific metrics
    total_chunks: int = 0
    total_bytes: int = 0
    total_audio_duration_ms: float = 0.0
    sentences_completed: int = 0

    # First chunk timing
    first_chunk_latency_ms: Optional[float] = None

    # Inter-chunk latency statistics
    avg_chunk_latency_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_min_ms: Optional[float] = None
    latency_max_ms: Optional[float] = None
    late_chunk_ratio: float = 0.0
    late_chunks: int = 0

    # Quality during stream
    stutters_detected: int = 0
    quality_drops: int = 0
    avg_stream_quality: float = 1.0

    # RTF
    streaming_rtf: float = 0.0

    # Link to full synthesis analytics
    synthesis_analytics: Optional[SynthesisAnalytics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/storage."""
        return {
            "session_id": self.session_id,
            "voice_id": self.voice_id,
            "timestamp": self.timestamp,
            "text_length": self.text_length,
            "total_chunks": self.total_chunks,
            "total_bytes": self.total_bytes,
            "total_audio_duration_ms": self.total_audio_duration_ms,
            "sentences_completed": self.sentences_completed,
            "first_chunk_latency_ms": self.first_chunk_latency_ms,
            "avg_chunk_latency_ms": self.avg_chunk_latency_ms,
            "latency_std_ms": self.latency_std_ms,
            "latency_min_ms": self.latency_min_ms,
            "latency_max_ms": self.latency_max_ms,
            "late_chunk_ratio": self.late_chunk_ratio,
            "late_chunks": self.late_chunks,
            "stutters_detected": self.stutters_detected,
            "quality_drops": self.quality_drops,
            "avg_stream_quality": self.avg_stream_quality,
            "streaming_rtf": self.streaming_rtf,
            "synthesis_analytics": (
                self.synthesis_analytics.to_dict()
                if self.synthesis_analytics else None
            ),
        }

    @classmethod
    def from_session_metrics(
        cls,
        session_metrics: Any,  # StreamSessionMetrics from streaming_collector
        text_length: int = 0,
        synthesis_analytics: Optional[SynthesisAnalytics] = None,
    ) -> "StreamingSessionAnalytics":
        """Create from StreamSessionMetrics."""
        return cls(
            session_id=session_metrics.session_id,
            voice_id=session_metrics.voice_id,
            timestamp=session_metrics.started_at,
            text_length=text_length,
            total_chunks=session_metrics.total_chunks,
            total_bytes=session_metrics.total_bytes,
            total_audio_duration_ms=session_metrics.total_audio_duration_ms,
            sentences_completed=session_metrics.sentences_completed,
            first_chunk_latency_ms=session_metrics.first_chunk_latency_ms,
            avg_chunk_latency_ms=session_metrics.avg_latency_ms,
            latency_std_ms=session_metrics.latency_std_ms,
            latency_min_ms=(
                session_metrics.latency_min_ms
                if session_metrics.latency_min_ms != float("inf") else None
            ),
            latency_max_ms=session_metrics.latency_max_ms,
            late_chunk_ratio=session_metrics.late_chunk_ratio,
            late_chunks=session_metrics.late_chunks,
            stutters_detected=session_metrics.stutters_detected,
            quality_drops=session_metrics.quality_drops,
            avg_stream_quality=session_metrics.get_current_quality_score(),
            streaming_rtf=session_metrics.real_time_factor,
            synthesis_analytics=synthesis_analytics,
        )


# ============================================================================
# AGGREGATE METRICS
# ============================================================================

@dataclass
class VoiceAggregateMetrics:
    """
    Aggregated metrics for a voice over a time period.

    Used for dashboard summaries and trend analysis.
    """
    voice_id: str
    period_start: datetime
    period_end: datetime

    # Volume
    total_syntheses: int = 0
    total_audio_seconds: float = 0.0
    total_characters: int = 0

    # Quality averages
    avg_mos_estimate: float = 0.0
    avg_naturalness: float = 0.0
    avg_snr_db: float = 0.0
    avg_quality_score: float = 0.0
    avg_reference_similarity: float = 0.0

    # Performance percentiles
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_real_time_factor: float = 0.0

    # Quality distribution
    quality_distribution: Dict[str, int] = field(default_factory=dict)

    # Issues
    error_count: int = 0
    governance_refusals: int = 0
    artifact_detections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "voice_id": self.voice_id,
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "volume": {
                "total_syntheses": self.total_syntheses,
                "total_audio_seconds": self.total_audio_seconds,
                "total_characters": self.total_characters,
            },
            "quality": {
                "avg_mos_estimate": self.avg_mos_estimate,
                "avg_naturalness": self.avg_naturalness,
                "avg_snr_db": self.avg_snr_db,
                "avg_quality_score": self.avg_quality_score,
                "avg_reference_similarity": self.avg_reference_similarity,
            },
            "performance": {
                "p50_latency_ms": self.p50_latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
                "p99_latency_ms": self.p99_latency_ms,
                "avg_real_time_factor": self.avg_real_time_factor,
            },
            "quality_distribution": self.quality_distribution,
            "issues": {
                "error_count": self.error_count,
                "governance_refusals": self.governance_refusals,
                "artifact_detections": self.artifact_detections,
            },
        }


@dataclass
class SystemAnalyticsSummary:
    """
    System-wide analytics summary.

    Aggregates metrics across all voices.
    """
    period_start: datetime
    period_end: datetime

    # Volume
    total_syntheses: int = 0
    total_audio_hours: float = 0.0
    unique_voices: int = 0
    unique_sessions: int = 0

    # Quality
    avg_mos_estimate: float = 0.0
    avg_naturalness: float = 0.0
    avg_snr_db: float = 0.0
    quality_score_p50: float = 0.0
    quality_score_p95: float = 0.0

    # Performance
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_rtf: float = 0.0

    # Errors
    total_errors: int = 0
    error_rate: float = 0.0
    governance_refusals: int = 0
    artifact_detections: int = 0

    # Top voices
    top_voices: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "volume": {
                "total_syntheses": self.total_syntheses,
                "total_audio_hours": self.total_audio_hours,
                "unique_voices": self.unique_voices,
                "unique_sessions": self.unique_sessions,
            },
            "quality": {
                "avg_mos_estimate": self.avg_mos_estimate,
                "avg_naturalness": self.avg_naturalness,
                "avg_snr_db": self.avg_snr_db,
                "quality_score_p50": self.quality_score_p50,
                "quality_score_p95": self.quality_score_p95,
            },
            "performance": {
                "avg_latency_ms": self.avg_latency_ms,
                "p50_latency_ms": self.p50_latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
                "p99_latency_ms": self.p99_latency_ms,
                "avg_rtf": self.avg_rtf,
            },
            "errors": {
                "total_errors": self.total_errors,
                "error_rate": self.error_rate,
                "governance_refusals": self.governance_refusals,
                "artifact_detections": self.artifact_detections,
            },
            "top_voices": self.top_voices,
        }


# ============================================================================
# ANALYTICS EVENTS
# ============================================================================

class AnalyticsEventType(str, Enum):
    """Types of analytics events for trend analysis."""
    QUALITY_DROP = "quality_drop"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_HIGH = "error_rate_high"
    ARTIFACT_DETECTED = "artifact_detected"
    VOICE_INCONSISTENCY = "voice_inconsistency"
    GOVERNANCE_REFUSAL = "governance_refusal"
    # Streaming events
    CHUNK_LATE = "chunk_late"
    STUTTER_DETECTED = "stutter_detected"
    STREAM_QUALITY_DROP = "stream_quality_drop"
    FIRST_CHUNK_SLOW = "first_chunk_slow"


class EventSeverity(str, Enum):
    """Severity levels for analytics events."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AnalyticsEvent:
    """
    Notable analytics event for alerting and trend analysis.
    """
    event_type: AnalyticsEventType
    timestamp: float
    voice_id: Optional[str] = None
    synthesis_id: Optional[str] = None
    severity: EventSeverity = EventSeverity.INFO
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "voice_id": self.voice_id,
            "synthesis_id": self.synthesis_id,
            "severity": self.severity.value,
            "message": self.message,
            "data": self.data,
            "resolved": self.resolved,
        }


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("  AXIOM VOX Analytics Models Demo")
    print("=" * 70)

    # Create sample technical quality metrics
    tech = TechnicalQualityMetrics(
        snr_db=28.5,
        peak_amplitude=0.85,
        rms_level_db=-18.0,
        dynamic_range_db=12.0,
        silence_ratio=0.05,
        clipping_samples=0,
        artifact_score=0.1,
    )
    print(f"\nTechnical Quality Score: {tech.get_quality_score():.2f}")

    # Create sample spectral metrics
    spectral = SpectralQualityMetrics(
        spectral_centroid_hz=1500.0,
        spectral_bandwidth_hz=800.0,
        spectral_rolloff_hz=4000.0,
        spectral_flatness=0.15,
        spectral_contrast=0.7,
        harmonic_ratio=0.85,
    )
    print(f"Spectral Quality Score: {spectral.get_quality_score():.2f}")

    # Create sample naturalness metrics
    naturalness = NaturalnessMetrics.from_mos(3.8)
    print(f"Naturalness (MOS {naturalness.mos_estimate}): {naturalness.overall_naturalness:.2f}")

    # Create sample performance metrics
    perf = PerformanceMetrics.compute(
        synthesis_latency_ms=250.0,
        audio_duration_ms=2500.0,
        text_length=100,
    )
    print(f"Real-time Factor: {perf.real_time_factor:.2f}")

    # Create complete synthesis analytics
    analytics = SynthesisAnalytics(
        synthesis_id="synth_abc123",
        voice_id="axiom_default",
        timestamp=time.time(),
        text_length=100,
        technical_quality=tech,
        spectral_quality=spectral,
        naturalness=naturalness,
        performance=perf,
    )

    print(f"\nComposite Quality Score: {analytics.get_quality_score():.2f}")
    print(f"Quality Tier: {analytics.get_quality_tier().value}")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
