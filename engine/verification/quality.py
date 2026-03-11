"""
VØX Verification - Quality Validation
--------------------------------------

Audio quality validation against thresholds.

AXIØM Phase 10: Verify - "How do we know this works?"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np

from .models import (
    QualityMetric,
    QualityResult,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityThresholds:
    """
    Quality thresholds for validation.

    Attributes:
        snr_min_db: Minimum signal-to-noise ratio in dB
        silence_max_ratio: Maximum ratio of silence
        clipping_max_ratio: Maximum ratio of clipped samples
        dc_offset_max: Maximum DC offset
        intelligibility_min: Minimum intelligibility score
    """
    snr_min_db: float = 20.0
    silence_max_ratio: float = 0.3
    clipping_max_ratio: float = 0.01
    dc_offset_max: float = 0.1
    intelligibility_min: float = 0.7
    energy_min_db: float = -40.0
    energy_max_db: float = 0.0
    spectral_flatness_max: float = 0.8


class QualityValidator:
    """
    Validator for audio quality.

    Features:
        - Signal-to-noise ratio (SNR)
        - Silence detection
        - Clipping detection
        - DC offset measurement
        - Spectral analysis
        - Intelligibility estimation
    """

    def __init__(
        self,
        thresholds: Optional[QualityThresholds] = None,
    ):
        """
        Initialize quality validator.

        Args:
            thresholds: Quality thresholds
        """
        self.thresholds = thresholds or QualityThresholds()

    def validate(
        self,
        audio: bytes,
        sample_rate: int = 24000,
        sample_width: int = 2,
    ) -> QualityResult:
        """
        Validate audio quality.

        Args:
            audio: Audio bytes
            sample_rate: Sample rate
            sample_width: Bytes per sample

        Returns:
            Quality validation result
        """
        start_time = time.time()

        # Convert to numpy array
        audio_array = self._bytes_to_array(audio, sample_width)

        if len(audio_array) == 0:
            return QualityResult(
                metrics=[],
                composite_score=0.0,
                audio_duration_ms=0.0,
                sample_rate=sample_rate,
                metadata={"error": "Empty audio"},
            )

        result = QualityResult(
            sample_rate=sample_rate,
            audio_duration_ms=len(audio_array) / sample_rate * 1000,
        )

        # Run quality checks
        self._check_energy(result, audio_array)
        self._check_snr(result, audio_array)
        self._check_silence(result, audio_array)
        self._check_clipping(result, audio_array)
        self._check_dc_offset(result, audio_array)
        self._check_spectral_flatness(result, audio_array, sample_rate)

        # Calculate composite score
        result.calculate_composite()

        result.analysis_duration_ms = (time.time() - start_time) * 1000
        return result

    def validate_streaming(
        self,
        chunks: List[bytes],
        sample_rate: int = 24000,
        sample_width: int = 2,
    ) -> QualityResult:
        """
        Validate quality of streaming audio.

        Args:
            chunks: List of audio chunks
            sample_rate: Sample rate
            sample_width: Bytes per sample

        Returns:
            Quality validation result
        """
        # Concatenate chunks
        audio = b"".join(chunks)
        result = self.validate(audio, sample_rate, sample_width)

        # Add streaming-specific metadata
        result.metadata["chunk_count"] = len(chunks)
        result.metadata["avg_chunk_size"] = len(audio) / len(chunks) if chunks else 0

        return result

    def check_threshold(
        self,
        metric_name: str,
        value: float,
    ) -> bool:
        """
        Check if a metric value passes threshold.

        Args:
            metric_name: Metric name
            value: Metric value

        Returns:
            True if passes threshold
        """
        threshold_map = {
            "snr_db": ("min", self.thresholds.snr_min_db),
            "silence_ratio": ("max", self.thresholds.silence_max_ratio),
            "clipping_ratio": ("max", self.thresholds.clipping_max_ratio),
            "dc_offset": ("max", self.thresholds.dc_offset_max),
            "intelligibility": ("min", self.thresholds.intelligibility_min),
            "energy_db": ("range", (self.thresholds.energy_min_db, self.thresholds.energy_max_db)),
        }

        if metric_name not in threshold_map:
            return True

        check_type, threshold = threshold_map[metric_name]

        if check_type == "min":
            return value >= threshold
        elif check_type == "max":
            return value <= threshold
        elif check_type == "range":
            return threshold[0] <= value <= threshold[1]

        return True

    def _bytes_to_array(
        self,
        audio: bytes,
        sample_width: int,
    ) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.int16

        audio_array = np.frombuffer(audio, dtype=dtype).astype(np.float32)

        # Normalize to [-1, 1]
        max_val = 2 ** (sample_width * 8 - 1)
        audio_array = audio_array / max_val

        return audio_array

    def _check_energy(
        self,
        result: QualityResult,
        audio: np.ndarray,
    ) -> None:
        """Check audio energy level."""
        # RMS energy in dB
        rms = np.sqrt(np.mean(audio ** 2))
        energy_db = 20 * np.log10(rms + 1e-10)

        result.metrics.append(QualityMetric(
            name="energy_db",
            value=float(energy_db),
            min_threshold=self.thresholds.energy_min_db,
            max_threshold=self.thresholds.energy_max_db,
            weight=1.0,
            description="RMS energy level in dB",
        ))

    def _check_snr(
        self,
        result: QualityResult,
        audio: np.ndarray,
    ) -> None:
        """Estimate signal-to-noise ratio."""
        # Simple SNR estimation using signal vs noise floor
        # Segment audio into frames
        frame_size = 1024
        n_frames = len(audio) // frame_size

        if n_frames < 2:
            return

        # Calculate RMS per frame
        frames = audio[:n_frames * frame_size].reshape(-1, frame_size)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # Estimate noise floor from quietest frames
        sorted_rms = np.sort(frame_rms)
        noise_floor = np.mean(sorted_rms[:max(1, n_frames // 10)])

        # Signal is mean of loudest frames
        signal_level = np.mean(sorted_rms[-max(1, n_frames // 10):])

        # Calculate SNR
        if noise_floor > 0:
            snr_db = 20 * np.log10(signal_level / (noise_floor + 1e-10))
        else:
            snr_db = 60.0  # Very high if no noise detected

        result.metrics.append(QualityMetric(
            name="snr_db",
            value=float(snr_db),
            min_threshold=self.thresholds.snr_min_db,
            weight=1.5,
            description="Estimated signal-to-noise ratio in dB",
        ))

    def _check_silence(
        self,
        result: QualityResult,
        audio: np.ndarray,
    ) -> None:
        """Check for excessive silence."""
        # Count samples below threshold
        silence_threshold = 0.01  # 1% of max amplitude
        silent_samples = np.sum(np.abs(audio) < silence_threshold)
        silence_ratio = silent_samples / len(audio)

        result.metrics.append(QualityMetric(
            name="silence_ratio",
            value=float(silence_ratio),
            max_threshold=self.thresholds.silence_max_ratio,
            weight=0.8,
            description="Ratio of silent samples",
        ))

    def _check_clipping(
        self,
        result: QualityResult,
        audio: np.ndarray,
    ) -> None:
        """Check for audio clipping."""
        # Count samples at or near max amplitude
        clip_threshold = 0.99
        clipped_samples = np.sum(np.abs(audio) >= clip_threshold)
        clipping_ratio = clipped_samples / len(audio)

        result.metrics.append(QualityMetric(
            name="clipping_ratio",
            value=float(clipping_ratio),
            max_threshold=self.thresholds.clipping_max_ratio,
            weight=1.2,
            description="Ratio of clipped samples",
        ))

    def _check_dc_offset(
        self,
        result: QualityResult,
        audio: np.ndarray,
    ) -> None:
        """Check for DC offset."""
        dc_offset = abs(float(np.mean(audio)))

        result.metrics.append(QualityMetric(
            name="dc_offset",
            value=dc_offset,
            max_threshold=self.thresholds.dc_offset_max,
            weight=0.5,
            description="DC offset (mean amplitude)",
        ))

    def _check_spectral_flatness(
        self,
        result: QualityResult,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Check spectral flatness (noise-like vs tonal)."""
        try:
            # Compute FFT
            n_fft = min(2048, len(audio))
            n_frames = len(audio) // n_fft

            if n_frames < 1:
                return

            flatness_values = []

            for i in range(n_frames):
                frame = audio[i * n_fft:(i + 1) * n_fft]
                spectrum = np.abs(np.fft.rfft(frame))

                # Spectral flatness = geometric mean / arithmetic mean
                geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
                arithmetic_mean = np.mean(spectrum)

                if arithmetic_mean > 0:
                    flatness = geometric_mean / arithmetic_mean
                    flatness_values.append(flatness)

            if flatness_values:
                avg_flatness = float(np.mean(flatness_values))

                result.metrics.append(QualityMetric(
                    name="spectral_flatness",
                    value=avg_flatness,
                    max_threshold=self.thresholds.spectral_flatness_max,
                    weight=0.7,
                    description="Spectral flatness (0=tonal, 1=noise)",
                ))
        except Exception as e:
            logger.warning(f"Spectral flatness check failed: {e}")


class IntelligibilityEstimator:
    """
    Estimate speech intelligibility.

    Uses simplified metrics without requiring a reference.
    """

    def __init__(self):
        """Initialize intelligibility estimator."""
        self.min_threshold = 0.7

    def estimate(
        self,
        audio: bytes,
        sample_rate: int = 24000,
        sample_width: int = 2,
    ) -> float:
        """
        Estimate intelligibility score.

        Args:
            audio: Audio bytes
            sample_rate: Sample rate
            sample_width: Bytes per sample

        Returns:
            Intelligibility score (0-1)
        """
        # Convert to array
        if sample_width == 2:
            dtype = np.int16
        else:
            dtype = np.int16

        audio_array = np.frombuffer(audio, dtype=dtype).astype(np.float32)
        audio_array = audio_array / (2 ** (sample_width * 8 - 1))

        if len(audio_array) == 0:
            return 0.0

        # Compute multiple intelligibility indicators
        scores = []

        # 1. Speech-weighted SNR
        snr_score = self._snr_score(audio_array)
        scores.append(snr_score * 0.3)

        # 2. Modulation depth (speech has natural modulation)
        mod_score = self._modulation_score(audio_array, sample_rate)
        scores.append(mod_score * 0.25)

        # 3. Spectral clarity
        clarity_score = self._spectral_clarity(audio_array, sample_rate)
        scores.append(clarity_score * 0.25)

        # 4. Dynamic range
        dynamic_score = self._dynamic_range_score(audio_array)
        scores.append(dynamic_score * 0.2)

        return min(1.0, sum(scores))

    def _snr_score(self, audio: np.ndarray) -> float:
        """Calculate SNR-based score."""
        frame_size = 1024
        n_frames = len(audio) // frame_size

        if n_frames < 2:
            return 0.5

        frames = audio[:n_frames * frame_size].reshape(-1, frame_size)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))

        sorted_rms = np.sort(frame_rms)
        noise_floor = np.mean(sorted_rms[:max(1, n_frames // 10)])
        signal_level = np.mean(sorted_rms[-max(1, n_frames // 10):])

        if noise_floor > 0:
            snr_db = 20 * np.log10(signal_level / (noise_floor + 1e-10))
        else:
            snr_db = 40.0

        # Map SNR to score (10dB = 0.5, 30dB = 1.0)
        return min(1.0, max(0.0, (snr_db - 10) / 20))

    def _modulation_score(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Calculate modulation-based score."""
        # Calculate envelope
        frame_size = int(sample_rate * 0.01)  # 10ms frames
        n_frames = len(audio) // frame_size

        if n_frames < 10:
            return 0.5

        frames = audio[:n_frames * frame_size].reshape(-1, frame_size)
        envelope = np.sqrt(np.mean(frames ** 2, axis=1))

        # Calculate modulation depth
        if np.max(envelope) > 0:
            mod_depth = (np.max(envelope) - np.min(envelope)) / np.max(envelope)
        else:
            mod_depth = 0

        # Speech typically has modulation depth 0.3-0.8
        if 0.3 <= mod_depth <= 0.8:
            return 1.0
        elif mod_depth < 0.3:
            return mod_depth / 0.3
        else:
            return max(0.5, 1.0 - (mod_depth - 0.8) / 0.2)

    def _spectral_clarity(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Calculate spectral clarity score."""
        n_fft = min(2048, len(audio))

        if len(audio) < n_fft:
            return 0.5

        spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
        freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)

        # Speech energy is concentrated in 80-4000 Hz
        speech_band = (freqs >= 80) & (freqs <= 4000)

        if np.sum(spectrum) > 0:
            speech_ratio = np.sum(spectrum[speech_band]) / np.sum(spectrum)
        else:
            speech_ratio = 0

        return min(1.0, speech_ratio * 1.5)

    def _dynamic_range_score(self, audio: np.ndarray) -> float:
        """Calculate dynamic range score."""
        frame_size = 1024
        n_frames = len(audio) // frame_size

        if n_frames < 2:
            return 0.5

        frames = audio[:n_frames * frame_size].reshape(-1, frame_size)
        frame_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10)

        dynamic_range = np.max(frame_db) - np.min(frame_db)

        # Speech typically has 20-40 dB dynamic range
        if 20 <= dynamic_range <= 40:
            return 1.0
        elif dynamic_range < 20:
            return dynamic_range / 20
        else:
            return max(0.5, 1.0 - (dynamic_range - 40) / 20)


def create_quality_validator(
    thresholds: Optional[Dict[str, float]] = None,
) -> QualityValidator:
    """
    Create quality validator with custom thresholds.

    Args:
        thresholds: Optional threshold overrides

    Returns:
        Configured quality validator
    """
    threshold_config = QualityThresholds()

    if thresholds:
        for key, value in thresholds.items():
            if hasattr(threshold_config, key):
                setattr(threshold_config, key, value)

    return QualityValidator(threshold_config)
