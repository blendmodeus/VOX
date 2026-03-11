"""
AXIOM VOX Metrics Collector
---------------------------

Synchronous metrics collection for synthesis operations.

Captures lightweight metrics inline during synthesis and
queues heavy computation for async processing.

Usage:
    from axiom_vox.analytics import VoxMetricsCollector

    collector = VoxMetricsCollector()

    # Start timing
    collector.start_synthesis("synth_123", "Hello world", "voice_1")

    # ... synthesis happens ...

    # End timing and compute metrics
    perf = collector.end_synthesis("synth_123")
    tech = collector.analyze_audio(audio_bytes, sample_rate)

    # Get complete analytics
    analytics = collector.get_analytics("synth_123")
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from queue import Queue
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from axiom_vox.analytics.models import (
    SynthesisAnalytics,
    TechnicalQualityMetrics,
    SpectralQualityMetrics,
    NaturalnessMetrics,
    PerformanceMetrics,
    ComputationStatus,
)
from axiom_vox.analytics.analyzer import VoxAudioAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SynthesisContext:
    """Context for a synthesis operation in progress."""
    synthesis_id: str
    voice_id: str
    text: str
    text_length: int
    start_time: float
    end_time: Optional[float] = None
    first_chunk_time: Optional[float] = None

    # Computed metrics
    audio_data: Optional[bytes] = None
    sample_rate: int = 24000
    audio_duration_ms: Optional[float] = None
    technical_quality: Optional[TechnicalQualityMetrics] = None
    spectral_quality: Optional[SpectralQualityMetrics] = None
    naturalness: Optional[NaturalnessMetrics] = None
    performance: Optional[PerformanceMetrics] = None

    # Governance info
    governance_action: Optional[str] = None
    content_emotion_match: Optional[float] = None
    manipulation_detected: bool = False


class VoxMetricsCollector:
    """
    Synchronous metrics collector for synthesis operations.

    Captures lightweight metrics inline during synthesis.
    Heavy computation (naturalness) is queued for async processing.
    """

    def __init__(
        self,
        enable_technical: bool = True,
        enable_spectral: bool = True,
        enable_naturalness: bool = False,  # Heavy, disabled by default
        async_queue: Optional[Queue] = None,
        max_contexts: int = 1000,
    ):
        """
        Initialize metrics collector.

        Args:
            enable_technical: Compute technical quality metrics
            enable_spectral: Compute spectral quality metrics
            enable_naturalness: Compute naturalness (heavy, async recommended)
            async_queue: Queue for heavy computation tasks
            max_contexts: Maximum synthesis contexts to keep in memory
        """
        self.enable_technical = enable_technical
        self.enable_spectral = enable_spectral
        self.enable_naturalness = enable_naturalness
        self.async_queue = async_queue
        self.max_contexts = max_contexts

        self._analyzer = VoxAudioAnalyzer()
        self._contexts: Dict[str, SynthesisContext] = {}
        self._lock = Lock()

    # ========================================================================
    # SYNTHESIS LIFECYCLE
    # ========================================================================

    def start_synthesis(
        self,
        synthesis_id: str,
        text: str,
        voice_id: str,
    ) -> None:
        """
        Mark synthesis start time.

        Args:
            synthesis_id: Unique identifier for this synthesis
            text: Text being synthesized
            voice_id: Voice ID being used
        """
        with self._lock:
            # Cleanup old contexts if needed
            if len(self._contexts) >= self.max_contexts:
                self._cleanup_old_contexts()

            self._contexts[synthesis_id] = SynthesisContext(
                synthesis_id=synthesis_id,
                voice_id=voice_id,
                text=text,
                text_length=len(text),
                start_time=time.time(),
            )

        logger.debug(f"Started synthesis tracking: {synthesis_id}")

    def record_first_chunk(self, synthesis_id: str) -> None:
        """
        Record time when first audio chunk is available (for streaming).

        Args:
            synthesis_id: Synthesis identifier
        """
        with self._lock:
            ctx = self._contexts.get(synthesis_id)
            if ctx and ctx.first_chunk_time is None:
                ctx.first_chunk_time = time.time()

    def end_synthesis(
        self,
        synthesis_id: str,
        audio_duration_ms: Optional[float] = None,
    ) -> PerformanceMetrics:
        """
        Mark synthesis end and compute performance metrics.

        Args:
            synthesis_id: Synthesis identifier
            audio_duration_ms: Duration of generated audio (if known)

        Returns:
            PerformanceMetrics instance
        """
        end_time = time.time()

        with self._lock:
            ctx = self._contexts.get(synthesis_id)
            if not ctx:
                logger.warning(f"No context found for synthesis: {synthesis_id}")
                return PerformanceMetrics(
                    synthesis_latency_ms=0,
                    audio_duration_ms=0,
                )

            ctx.end_time = end_time
            if audio_duration_ms is not None:
                ctx.audio_duration_ms = audio_duration_ms

            # Compute performance metrics
            latency_ms = (end_time - ctx.start_time) * 1000
            first_chunk_latency = None
            if ctx.first_chunk_time:
                first_chunk_latency = (ctx.first_chunk_time - ctx.start_time) * 1000

            ctx.performance = PerformanceMetrics.compute(
                synthesis_latency_ms=latency_ms,
                audio_duration_ms=ctx.audio_duration_ms or 0,
                text_length=ctx.text_length,
                first_chunk_latency_ms=first_chunk_latency,
            )

            logger.debug(f"Ended synthesis tracking: {synthesis_id}, latency={latency_ms:.1f}ms")
            return ctx.performance

    # ========================================================================
    # AUDIO ANALYSIS
    # ========================================================================

    def analyze_audio(
        self,
        synthesis_id: str,
        audio_data: bytes,
        sample_rate: int = 24000,
    ) -> Tuple[Optional[TechnicalQualityMetrics], Optional[SpectralQualityMetrics]]:
        """
        Analyze audio quality (synchronous).

        Args:
            synthesis_id: Synthesis identifier
            audio_data: Raw audio bytes
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (technical_metrics, spectral_metrics)
        """
        with self._lock:
            ctx = self._contexts.get(synthesis_id)

        if not ctx:
            logger.warning(f"No context found for synthesis: {synthesis_id}")
            return None, None

        # Convert audio to array
        audio_array = self._analyzer.bytes_to_array(audio_data, sample_rate)
        if audio_array is None:
            return None, None

        # Compute audio duration if not already set
        with self._lock:
            if ctx.audio_duration_ms is None:
                ctx.audio_duration_ms = len(audio_array) / sample_rate * 1000
                ctx.audio_data = audio_data
                ctx.sample_rate = sample_rate

        # Technical quality metrics
        tech = None
        if self.enable_technical:
            tech = self._analyzer.compute_technical_metrics(audio_array, sample_rate)
            with self._lock:
                ctx.technical_quality = tech

        # Spectral quality metrics
        spectral = None
        if self.enable_spectral:
            spectral = self._analyzer.compute_spectral_metrics(audio_array, sample_rate)
            with self._lock:
                ctx.spectral_quality = spectral

        # Queue naturalness for async if enabled
        if self.enable_naturalness and self.async_queue:
            self.queue_naturalness_analysis(synthesis_id, audio_data, sample_rate)

        return tech, spectral

    def queue_naturalness_analysis(
        self,
        synthesis_id: str,
        audio_data: bytes,
        sample_rate: int = 24000,
    ) -> None:
        """
        Queue heavy naturalness computation for async processing.

        Args:
            synthesis_id: Synthesis identifier
            audio_data: Raw audio bytes
            sample_rate: Sample rate in Hz
        """
        if self.async_queue is None:
            logger.warning("No async queue configured, computing naturalness synchronously")
            self.compute_naturalness_sync(synthesis_id, audio_data, sample_rate)
            return

        task = {
            "type": "naturalness",
            "synthesis_id": synthesis_id,
            "audio_data": audio_data,
            "sample_rate": sample_rate,
        }
        self.async_queue.put(task)
        logger.debug(f"Queued naturalness analysis for: {synthesis_id}")

    def compute_naturalness_sync(
        self,
        synthesis_id: str,
        audio_data: bytes,
        sample_rate: int = 24000,
    ) -> Optional[NaturalnessMetrics]:
        """
        Compute naturalness metrics synchronously (may be slow).

        Args:
            synthesis_id: Synthesis identifier
            audio_data: Raw audio bytes
            sample_rate: Sample rate in Hz

        Returns:
            NaturalnessMetrics instance
        """
        audio_array = self._analyzer.bytes_to_array(audio_data, sample_rate)
        if audio_array is None:
            return None

        naturalness = self._analyzer.estimate_naturalness(audio_array, sample_rate)

        with self._lock:
            ctx = self._contexts.get(synthesis_id)
            if ctx:
                ctx.naturalness = naturalness

        return naturalness

    # ========================================================================
    # GOVERNANCE INTEGRATION
    # ========================================================================

    def record_governance(
        self,
        synthesis_id: str,
        action: str,
        content_emotion_match: Optional[float] = None,
        manipulation_detected: bool = False,
    ) -> None:
        """
        Record governance decision for a synthesis.

        Args:
            synthesis_id: Synthesis identifier
            action: Governance action (allow/repair/refuse)
            content_emotion_match: Content-emotion match score
            manipulation_detected: Whether manipulation was detected
        """
        with self._lock:
            ctx = self._contexts.get(synthesis_id)
            if ctx:
                ctx.governance_action = action
                ctx.content_emotion_match = content_emotion_match
                ctx.manipulation_detected = manipulation_detected

    # ========================================================================
    # ANALYTICS RETRIEVAL
    # ========================================================================

    def get_analytics(self, synthesis_id: str) -> Optional[SynthesisAnalytics]:
        """
        Get complete analytics for a synthesis.

        Args:
            synthesis_id: Synthesis identifier

        Returns:
            SynthesisAnalytics instance, or None if not found
        """
        with self._lock:
            ctx = self._contexts.get(synthesis_id)

        if not ctx:
            return None

        # Determine computation status
        if ctx.naturalness is not None:
            status = ComputationStatus.COMPLETE
        elif ctx.technical_quality is not None or ctx.spectral_quality is not None:
            status = ComputationStatus.PENDING  # Still waiting for naturalness
        else:
            status = ComputationStatus.PENDING

        return SynthesisAnalytics(
            synthesis_id=ctx.synthesis_id,
            voice_id=ctx.voice_id,
            timestamp=ctx.start_time,
            text_length=ctx.text_length,
            technical_quality=ctx.technical_quality,
            spectral_quality=ctx.spectral_quality,
            naturalness=ctx.naturalness,
            performance=ctx.performance,
            governance_action=ctx.governance_action,
            content_emotion_match=ctx.content_emotion_match,
            manipulation_detected=ctx.manipulation_detected,
            computation_status=status,
        )

    def update_naturalness(
        self,
        synthesis_id: str,
        naturalness: NaturalnessMetrics,
    ) -> None:
        """
        Update naturalness metrics (called by async worker).

        Args:
            synthesis_id: Synthesis identifier
            naturalness: Computed naturalness metrics
        """
        with self._lock:
            ctx = self._contexts.get(synthesis_id)
            if ctx:
                ctx.naturalness = naturalness

    def remove_context(self, synthesis_id: str) -> Optional[SynthesisAnalytics]:
        """
        Remove and return analytics for a synthesis.

        Use this after storing analytics to free memory.

        Args:
            synthesis_id: Synthesis identifier

        Returns:
            Final SynthesisAnalytics, or None if not found
        """
        analytics = self.get_analytics(synthesis_id)

        with self._lock:
            self._contexts.pop(synthesis_id, None)

        return analytics

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _cleanup_old_contexts(self) -> None:
        """Remove oldest contexts when limit is reached."""
        if len(self._contexts) < self.max_contexts:
            return

        # Sort by start time and remove oldest 10%
        sorted_ids = sorted(
            self._contexts.keys(),
            key=lambda k: self._contexts[k].start_time,
        )
        remove_count = max(1, len(sorted_ids) // 10)

        for synthesis_id in sorted_ids[:remove_count]:
            self._contexts.pop(synthesis_id, None)

        logger.debug(f"Cleaned up {remove_count} old synthesis contexts")

    def generate_synthesis_id(self) -> str:
        """Generate a unique synthesis ID."""
        return f"synth_{uuid.uuid4().hex[:12]}"

    def get_active_count(self) -> int:
        """Get count of active synthesis contexts."""
        with self._lock:
            return len(self._contexts)


# ============================================================================
# GLOBAL COLLECTOR INSTANCE
# ============================================================================

_default_collector: Optional[VoxMetricsCollector] = None


def get_collector() -> VoxMetricsCollector:
    """Get or create the default metrics collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = VoxMetricsCollector()
    return _default_collector


def set_collector(collector: VoxMetricsCollector) -> None:
    """Set the default metrics collector."""
    global _default_collector
    _default_collector = collector


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    import struct

    print("=" * 70)
    print("  AXIOM VOX Metrics Collector Demo")
    print("=" * 70)

    collector = VoxMetricsCollector(
        enable_technical=True,
        enable_spectral=True,
        enable_naturalness=True,  # Sync for demo
    )

    # Simulate synthesis
    synthesis_id = collector.generate_synthesis_id()
    text = "Hello, this is a test of the voice analytics system."

    print(f"\nSynthesis ID: {synthesis_id}")
    print(f"Text: {text}")
    print("-" * 70)

    # Start synthesis
    collector.start_synthesis(synthesis_id, text, "test_voice")
    print("Started synthesis tracking...")

    # Simulate synthesis delay
    time.sleep(0.25)

    # End synthesis
    perf = collector.end_synthesis(synthesis_id, audio_duration_ms=2500)
    print(f"\nPerformance Metrics:")
    print(f"  Latency: {perf.synthesis_latency_ms:.1f} ms")
    print(f"  Audio Duration: {perf.audio_duration_ms:.1f} ms")
    print(f"  RTF: {perf.real_time_factor:.3f}")

    # Create dummy audio for analysis
    try:
        import numpy as np
        sample_rate = 24000
        duration = 2.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()

        # Analyze audio
        tech, spectral = collector.analyze_audio(synthesis_id, audio_bytes, sample_rate)

        if tech:
            print(f"\nTechnical Quality:")
            print(f"  SNR: {tech.snr_db:.1f} dB")
            print(f"  Quality Score: {tech.get_quality_score():.2f}")

        if spectral:
            print(f"\nSpectral Quality:")
            print(f"  Centroid: {spectral.spectral_centroid_hz:.0f} Hz")
            print(f"  Quality Score: {spectral.get_quality_score():.2f}")

        # Compute naturalness (sync for demo)
        naturalness = collector.compute_naturalness_sync(synthesis_id, audio_bytes, sample_rate)
        if naturalness:
            print(f"\nNaturalness:")
            print(f"  MOS: {naturalness.mos_estimate:.2f}")
            print(f"  Overall: {naturalness.overall_naturalness:.2f}")

    except ImportError:
        print("\nnumpy not available - skipping audio analysis")

    # Record governance
    collector.record_governance(synthesis_id, "allow", content_emotion_match=0.85)

    # Get complete analytics
    analytics = collector.get_analytics(synthesis_id)
    if analytics:
        print(f"\nComplete Analytics:")
        print(f"  Quality Score: {analytics.get_quality_score():.2f}")
        print(f"  Quality Tier: {analytics.get_quality_tier().value}")
        print(f"  Status: {analytics.computation_status.value}")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
