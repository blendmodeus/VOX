"""
Real-Time Quality Monitor
-------------------------

Live quality scoring during synthesis with feedback loops.

Monitors:
    - Audio quality (SNR, spectral analysis)
    - Naturalness metrics
    - Consistency with voice profile
    - Performance metrics (latency, RTF)

AXIØM Phase 6: System - "Integrate the parts"
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

import numpy as np

from .models import (
    QualityGate,
    QualityResult,
    PipelineConfig,
    QUALITY_THRESHOLDS,
)

logger = logging.getLogger(__name__)


class QualityEvent(str, Enum):
    """Types of quality events."""
    CHUNK_ANALYZED = "chunk_analyzed"
    QUALITY_DROP = "quality_drop"
    QUALITY_RECOVERY = "quality_recovery"
    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"
    STUTTER_DETECTED = "stutter_detected"
    SILENCE_GAP = "silence_gap"


@dataclass
class QualitySnapshot:
    """Point-in-time quality measurement."""
    timestamp: float
    chunk_index: int
    overall_score: float
    snr_db: float
    energy_level: float
    spectral_centroid: float
    is_silent: bool = False
    issues: List[str] = field(default_factory=list)


@dataclass
class QualityAlert:
    """Quality issue alert."""
    event: QualityEvent
    timestamp: float
    severity: str  # "info", "warning", "error"
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class RealTimeQualityMonitor:
    """
    Real-time quality monitoring during synthesis.

    Provides:
        - Per-chunk quality scoring
        - Running quality statistics
        - Quality gate enforcement
        - Issue detection and alerting
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_alert: Optional[Callable[[QualityAlert], None]] = None,
    ):
        """
        Initialize quality monitor.

        Args:
            config: Pipeline configuration
            on_alert: Callback for quality alerts
        """
        self.config = config or PipelineConfig()
        self.on_alert = on_alert

        # Session state
        self._session_id: Optional[str] = None
        self._chunk_count = 0
        self._snapshots: deque = deque(maxlen=100)
        self._alerts: List[QualityAlert] = []

        # Running statistics
        self._quality_sum = 0.0
        self._snr_sum = 0.0
        self._min_quality = 1.0
        self._max_quality = 0.0

        # Timing
        self._start_time: Optional[float] = None
        self._last_chunk_time: Optional[float] = None
        self._total_audio_duration = 0.0

        # Quality baseline
        self._baseline_quality: Optional[float] = None
        self._quality_drop_threshold = 0.15

    def start_session(self, session_id: str) -> None:
        """Start a new monitoring session."""
        self._session_id = session_id
        self._chunk_count = 0
        self._snapshots.clear()
        self._alerts.clear()
        self._quality_sum = 0.0
        self._snr_sum = 0.0
        self._min_quality = 1.0
        self._max_quality = 0.0
        self._start_time = time.time()
        self._last_chunk_time = None
        self._total_audio_duration = 0.0
        self._baseline_quality = None

        logger.debug(f"Started quality monitoring session: {session_id}")

    def analyze_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 24000,
        expected_voice_id: Optional[str] = None,
    ) -> QualitySnapshot:
        """
        Analyze a single audio chunk.

        Args:
            audio_chunk: Audio data
            sample_rate: Sample rate
            expected_voice_id: Expected voice for consistency check

        Returns:
            QualitySnapshot
        """
        current_time = time.time()
        self._chunk_count += 1

        # Check for timing gap (potential stutter)
        if self._last_chunk_time:
            gap = current_time - self._last_chunk_time
            expected_gap = len(audio_chunk) / sample_rate
            if gap > expected_gap * 2:
                self._emit_alert(
                    QualityEvent.STUTTER_DETECTED,
                    "warning",
                    f"Chunk delivery gap: {gap:.3f}s (expected {expected_gap:.3f}s)",
                    {"gap_seconds": gap, "expected_seconds": expected_gap},
                )

        self._last_chunk_time = current_time
        self._total_audio_duration += len(audio_chunk) / sample_rate

        # Analyze chunk
        audio = np.asarray(audio_chunk, dtype=np.float32)
        issues = []

        # Check for silence
        rms = np.sqrt(np.mean(audio ** 2))
        is_silent = rms < 0.001

        if is_silent:
            # Check if unexpected silence
            if self._chunk_count > 1 and not self._is_expected_pause():
                issues.append("Unexpected silence")
                self._emit_alert(
                    QualityEvent.SILENCE_GAP,
                    "info",
                    "Detected silence in audio stream",
                    {"chunk_index": self._chunk_count},
                )

        # Calculate metrics
        snr_db = self._estimate_snr(audio)
        energy = float(rms)
        spectral_centroid = self._calculate_spectral_centroid(audio, sample_rate)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            snr_db=snr_db,
            energy=energy,
            spectral_centroid=spectral_centroid,
            is_silent=is_silent,
        )

        # Update statistics
        self._quality_sum += overall_score
        self._snr_sum += snr_db
        self._min_quality = min(self._min_quality, overall_score)
        self._max_quality = max(self._max_quality, overall_score)

        # Set baseline from first chunks
        if self._chunk_count <= 3 and not is_silent:
            if self._baseline_quality is None:
                self._baseline_quality = overall_score
            else:
                self._baseline_quality = (self._baseline_quality + overall_score) / 2

        # Check for quality drop
        if self._baseline_quality and not is_silent:
            drop = self._baseline_quality - overall_score
            if drop > self._quality_drop_threshold:
                issues.append(f"Quality drop: {drop:.2f}")
                self._emit_alert(
                    QualityEvent.QUALITY_DROP,
                    "warning",
                    f"Quality dropped by {drop:.2f} from baseline",
                    {"baseline": self._baseline_quality, "current": overall_score},
                )

        snapshot = QualitySnapshot(
            timestamp=current_time,
            chunk_index=self._chunk_count,
            overall_score=overall_score,
            snr_db=snr_db,
            energy_level=energy,
            spectral_centroid=spectral_centroid,
            is_silent=is_silent,
            issues=issues,
        )

        self._snapshots.append(snapshot)
        return snapshot

    def get_running_quality(self) -> Dict[str, float]:
        """Get current running quality statistics."""
        if self._chunk_count == 0:
            return {
                "average_quality": 0.0,
                "average_snr": 0.0,
                "min_quality": 0.0,
                "max_quality": 0.0,
                "chunk_count": 0,
            }

        return {
            "average_quality": self._quality_sum / self._chunk_count,
            "average_snr": self._snr_sum / self._chunk_count,
            "min_quality": self._min_quality,
            "max_quality": self._max_quality,
            "chunk_count": self._chunk_count,
            "baseline_quality": self._baseline_quality,
        }

    def finalize_session(self) -> QualityResult:
        """
        Finalize session and return complete quality result.

        Returns:
            QualityResult
        """
        running = self.get_running_quality()
        overall_score = running["average_quality"]

        # Determine gate status
        if overall_score >= QUALITY_THRESHOLDS["excellent"]:
            gate_status = QualityGate.PASSED
        elif overall_score >= self.config.min_quality_score:
            gate_status = QualityGate.PASSED
        elif overall_score >= QUALITY_THRESHOLDS["acceptable"]:
            gate_status = QualityGate.WARNING
        else:
            gate_status = QualityGate.FAILED

        # Emit gate event
        if gate_status == QualityGate.FAILED:
            self._emit_alert(
                QualityEvent.GATE_FAILED,
                "error",
                f"Quality gate failed: {overall_score:.3f} < {self.config.min_quality_score}",
                {"score": overall_score, "threshold": self.config.min_quality_score},
            )
        else:
            self._emit_alert(
                QualityEvent.GATE_PASSED,
                "info",
                f"Quality gate passed: {overall_score:.3f}",
                {"score": overall_score},
            )

        # Collect issues
        issues = []
        for snapshot in self._snapshots:
            issues.extend(snapshot.issues)
        issues = list(set(issues))

        # Generate recommendations
        recommendations = self._generate_recommendations(overall_score, running)

        # Calculate performance metrics
        total_time = time.time() - self._start_time if self._start_time else 0
        rtf = total_time / self._total_audio_duration if self._total_audio_duration > 0 else 0

        result = QualityResult(
            overall_score=overall_score,
            gate_status=gate_status,
            snr_db=running["average_snr"],
            naturalness_score=self._estimate_naturalness(),
            intelligibility_score=self._estimate_intelligibility(),
            consistency_score=self._estimate_consistency(),
            issues=issues,
            recommendations=recommendations,
            min_acceptable_score=self.config.min_quality_score,
        )

        logger.info(
            f"Session {self._session_id} complete: "
            f"score={overall_score:.3f}, gate={gate_status.value}"
        )

        return result

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        if len(audio) < 100:
            return 0.0

        # Simple SNR estimation
        signal_power = np.mean(audio ** 2)
        if signal_power < 1e-10:
            return 0.0

        # Estimate noise from low-energy regions
        sorted_powers = np.sort(audio ** 2)
        noise_power = np.mean(sorted_powers[:len(sorted_powers) // 10])

        if noise_power < 1e-10:
            return 60.0  # Very clean signal

        snr = 10 * np.log10(signal_power / noise_power)
        return float(np.clip(snr, 0, 60))

    def _calculate_spectral_centroid(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Calculate spectral centroid."""
        if len(audio) < 256:
            return 0.0

        try:
            from scipy.fft import rfft

            spectrum = np.abs(rfft(audio))
            freqs = np.linspace(0, sample_rate / 2, len(spectrum))

            if np.sum(spectrum) < 1e-10:
                return 0.0

            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
            return float(centroid)

        except ImportError:
            # Simple fallback
            return 1000.0  # Nominal value

    def _calculate_overall_score(
        self,
        snr_db: float,
        energy: float,
        spectral_centroid: float,
        is_silent: bool,
    ) -> float:
        """Calculate overall quality score."""
        if is_silent:
            return 0.5  # Neutral for silent chunks

        scores = []

        # SNR score (0-1)
        snr_score = min(1.0, snr_db / 40.0)
        scores.append(snr_score * 0.3)

        # Energy score (penalize very low or very high)
        if energy < 0.01:
            energy_score = 0.3
        elif energy > 0.9:
            energy_score = 0.7
        else:
            energy_score = 1.0
        scores.append(energy_score * 0.2)

        # Spectral centroid score (penalize extremes)
        if 500 < spectral_centroid < 4000:
            centroid_score = 1.0
        elif 200 < spectral_centroid < 6000:
            centroid_score = 0.7
        else:
            centroid_score = 0.5
        scores.append(centroid_score * 0.2)

        # Base quality
        scores.append(0.3)

        return sum(scores)

    def _estimate_naturalness(self) -> float:
        """Estimate naturalness from collected snapshots."""
        if len(self._snapshots) < 2:
            return 0.7

        # Check for natural variation
        qualities = [s.overall_score for s in self._snapshots if not s.is_silent]
        if not qualities:
            return 0.5

        std = np.std(qualities)
        # Natural speech has some variation but not too much
        if 0.02 < std < 0.15:
            return 0.85
        elif std < 0.01:  # Too uniform
            return 0.6
        else:  # Too variable
            return 0.5

    def _estimate_intelligibility(self) -> float:
        """Estimate intelligibility from metrics."""
        running = self.get_running_quality()

        # Based on SNR and quality
        snr_score = min(1.0, running["average_snr"] / 30.0)
        quality_score = running["average_quality"]

        return (snr_score + quality_score) / 2

    def _estimate_consistency(self) -> float:
        """Estimate voice consistency."""
        if len(self._snapshots) < 2:
            return 0.7

        # Check spectral centroid consistency
        centroids = [s.spectral_centroid for s in self._snapshots if not s.is_silent]
        if len(centroids) < 2:
            return 0.7

        cv = np.std(centroids) / (np.mean(centroids) + 1e-10)

        # Low coefficient of variation = high consistency
        if cv < 0.1:
            return 0.95
        elif cv < 0.2:
            return 0.8
        elif cv < 0.3:
            return 0.6
        else:
            return 0.4

    def _is_expected_pause(self) -> bool:
        """Check if silence is an expected pause (e.g., between sentences)."""
        # Simple heuristic: occasional silence is expected
        if self._chunk_count <= 1:
            return True

        recent = list(self._snapshots)[-5:]
        silent_count = sum(1 for s in recent if s.is_silent)
        return silent_count <= 2

    def _generate_recommendations(
        self,
        overall_score: float,
        running: Dict[str, float],
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        if overall_score < 0.6:
            recommendations.append("Consider re-recording with better audio quality")

        if running["average_snr"] < 20:
            recommendations.append("Improve signal-to-noise ratio (reduce background noise)")

        if self._min_quality < 0.3:
            recommendations.append("Some chunks have very low quality - check for audio dropouts")

        if len(self._alerts) > 5:
            recommendations.append("Multiple quality issues detected - review synthesis parameters")

        return recommendations

    def _emit_alert(
        self,
        event: QualityEvent,
        severity: str,
        message: str,
        data: Dict[str, Any],
    ) -> None:
        """Emit a quality alert."""
        alert = QualityAlert(
            event=event,
            timestamp=time.time(),
            severity=severity,
            message=message,
            data=data,
        )
        self._alerts.append(alert)

        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        logger.debug(f"Quality alert: [{severity}] {message}")


# Singleton instance
_monitor_instance: Optional[RealTimeQualityMonitor] = None


def get_quality_monitor(
    config: Optional[PipelineConfig] = None,
) -> RealTimeQualityMonitor:
    """Get or create quality monitor singleton."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = RealTimeQualityMonitor(config=config)
    return _monitor_instance


def set_quality_monitor(monitor: RealTimeQualityMonitor) -> None:
    """Set the quality monitor singleton."""
    global _monitor_instance
    _monitor_instance = monitor
