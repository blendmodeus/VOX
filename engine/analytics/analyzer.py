"""
AXIOM VOX Audio Analyzer
------------------------

Audio quality analysis utilities for voice analytics.

Components:
- SNR estimation
- Spectral analysis (centroid, flatness, rolloff)
- Artifact detection
- Naturalness estimation (heuristic-based)

Uses numpy and scipy for signal processing.
Heavy computation (full spectral analysis, naturalness) can be run async.

Usage:
    from axiom_vox.analytics import VoxAudioAnalyzer

    analyzer = VoxAudioAnalyzer()
    snr = analyzer.compute_snr(audio_array, sample_rate)
    spectral = analyzer.compute_spectral_metrics(audio_array, sample_rate)
"""

import logging
import math
from typing import List, Optional, Tuple, Any
import struct

from axiom_vox.analytics.models import (
    TechnicalQualityMetrics,
    SpectralQualityMetrics,
    NaturalnessMetrics,
)

logger = logging.getLogger(__name__)

# Try to import numpy/scipy - graceful fallback if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    logger.warning("numpy not available - audio analysis will be limited")

try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    signal = None
    fft = None
    fftfreq = None
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - spectral analysis will be limited")


class VoxAudioAnalyzer:
    """
    Audio quality analysis for voice synthesis.

    Computes technical quality metrics, spectral characteristics,
    and naturalness estimates from audio waveforms.
    """

    def __init__(
        self,
        silence_threshold_db: float = -40.0,
        clipping_threshold: float = 0.99,
    ):
        """
        Initialize analyzer.

        Args:
            silence_threshold_db: Threshold below which audio is considered silence
            clipping_threshold: Amplitude above which samples are considered clipped
        """
        self.silence_threshold_db = silence_threshold_db
        self.clipping_threshold = clipping_threshold
        self._silence_threshold_linear = 10 ** (silence_threshold_db / 20)

    # ========================================================================
    # AUDIO CONVERSION
    # ========================================================================

    def bytes_to_array(
        self,
        audio_data: bytes,
        sample_rate: int = 24000,
        sample_width: int = 2,
    ) -> Optional[Any]:
        """
        Convert audio bytes to numpy array.

        Args:
            audio_data: Raw audio bytes (PCM)
            sample_rate: Sample rate in Hz
            sample_width: Bytes per sample (2 = 16-bit)

        Returns:
            Numpy array of float samples (-1 to 1), or None if numpy unavailable
        """
        if not NUMPY_AVAILABLE:
            return None

        # Skip WAV header if present
        if audio_data[:4] == b'RIFF':
            # Find data chunk
            pos = 12
            while pos < len(audio_data) - 8:
                chunk_id = audio_data[pos:pos+4]
                chunk_size = struct.unpack('<I', audio_data[pos+4:pos+8])[0]
                if chunk_id == b'data':
                    audio_data = audio_data[pos+8:pos+8+chunk_size]
                    break
                pos += 8 + chunk_size

        # Convert to numpy array
        if sample_width == 2:
            samples = np.frombuffer(audio_data, dtype=np.int16)
            return samples.astype(np.float32) / 32768.0
        elif sample_width == 4:
            samples = np.frombuffer(audio_data, dtype=np.int32)
            return samples.astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(audio_data, dtype=np.uint8)
            return (samples.astype(np.float32) - 128) / 128.0

    # ========================================================================
    # TECHNICAL QUALITY METRICS
    # ========================================================================

    def compute_snr(
        self,
        audio: Any,
        sample_rate: int = 24000,
    ) -> float:
        """
        Estimate signal-to-noise ratio.

        Uses a simple energy-based approach:
        - Signal: RMS of non-silent segments
        - Noise: RMS of silent segments (or estimated floor)

        Args:
            audio: Numpy array of audio samples
            sample_rate: Sample rate in Hz

        Returns:
            SNR in dB
        """
        if not NUMPY_AVAILABLE or audio is None:
            return 20.0  # Default reasonable value

        # Compute RMS in short windows
        window_size = int(sample_rate * 0.02)  # 20ms windows
        hop_size = window_size // 2

        rms_values = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)

        if not rms_values:
            return 20.0

        rms_values = np.array(rms_values)

        # Separate signal and noise based on threshold
        signal_mask = rms_values > self._silence_threshold_linear
        noise_mask = ~signal_mask

        if np.sum(signal_mask) == 0:
            return 0.0  # No signal

        signal_rms = np.mean(rms_values[signal_mask])

        if np.sum(noise_mask) > 0:
            noise_rms = np.mean(rms_values[noise_mask])
        else:
            # Estimate noise floor from lowest 10% of values
            noise_rms = np.percentile(rms_values, 10)

        if noise_rms < 1e-10:
            noise_rms = 1e-10  # Prevent division by zero

        snr = 20 * np.log10(signal_rms / noise_rms)
        return float(min(60.0, max(0.0, snr)))  # Clamp to reasonable range

    def compute_technical_metrics(
        self,
        audio: Any,
        sample_rate: int = 24000,
    ) -> TechnicalQualityMetrics:
        """
        Compute all technical quality metrics.

        Args:
            audio: Numpy array of audio samples
            sample_rate: Sample rate in Hz

        Returns:
            TechnicalQualityMetrics instance
        """
        if not NUMPY_AVAILABLE or audio is None:
            # Return default values
            return TechnicalQualityMetrics(
                snr_db=20.0,
                peak_amplitude=0.5,
                rms_level_db=-20.0,
                dynamic_range_db=12.0,
                silence_ratio=0.1,
                clipping_samples=0,
                artifact_score=0.1,
            )

        # Peak amplitude
        peak = float(np.max(np.abs(audio)))

        # RMS level
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Dynamic range
        peak_db = 20 * np.log10(peak + 1e-10)
        dynamic_range = peak_db - rms_db

        # Silence ratio
        silence_samples = np.sum(np.abs(audio) < self._silence_threshold_linear)
        silence_ratio = silence_samples / len(audio)

        # Clipping detection
        clipping_samples = int(np.sum(np.abs(audio) > self.clipping_threshold))

        # SNR
        snr = self.compute_snr(audio, sample_rate)

        # Artifact score (heuristic based on sudden amplitude changes)
        artifact_score = self._detect_artifact_score(audio, sample_rate)

        return TechnicalQualityMetrics(
            snr_db=snr,
            peak_amplitude=peak,
            rms_level_db=float(rms_db),
            dynamic_range_db=float(dynamic_range),
            silence_ratio=float(silence_ratio),
            clipping_samples=clipping_samples,
            artifact_score=artifact_score,
        )

    def _detect_artifact_score(
        self,
        audio: Any,
        sample_rate: int,
    ) -> float:
        """
        Detect audio artifacts (clicks, pops, glitches).

        Returns score from 0 (no artifacts) to 1 (severe artifacts).
        """
        if not NUMPY_AVAILABLE:
            return 0.1

        # Compute derivative (sudden changes indicate artifacts)
        diff = np.abs(np.diff(audio))

        # Find outliers in derivative (potential artifacts)
        threshold = np.mean(diff) + 4 * np.std(diff)
        artifacts = np.sum(diff > threshold)

        # Normalize by audio length
        artifact_ratio = artifacts / len(audio)

        # Map to 0-1 score (more artifacts = higher score)
        return float(min(1.0, artifact_ratio * 1000))

    # ========================================================================
    # SPECTRAL ANALYSIS
    # ========================================================================

    def compute_spectral_metrics(
        self,
        audio: Any,
        sample_rate: int = 24000,
    ) -> SpectralQualityMetrics:
        """
        Compute spectral quality metrics.

        Args:
            audio: Numpy array of audio samples
            sample_rate: Sample rate in Hz

        Returns:
            SpectralQualityMetrics instance
        """
        if not NUMPY_AVAILABLE or not SCIPY_AVAILABLE or audio is None:
            # Return default values
            return SpectralQualityMetrics(
                spectral_centroid_hz=1500.0,
                spectral_bandwidth_hz=800.0,
                spectral_rolloff_hz=4000.0,
                spectral_flatness=0.2,
                spectral_contrast=0.6,
                harmonic_ratio=0.8,
            )

        # Compute magnitude spectrum
        n_fft = min(2048, len(audio))
        spectrum = np.abs(fft(audio[:n_fft]))[:n_fft // 2]
        freqs = fftfreq(n_fft, 1 / sample_rate)[:n_fft // 2]

        # Normalize spectrum
        spectrum_sum = np.sum(spectrum) + 1e-10

        # Spectral centroid (weighted mean frequency)
        centroid = np.sum(freqs * spectrum) / spectrum_sum

        # Spectral bandwidth (weighted std of frequencies)
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spectrum_sum)

        # Spectral rolloff (frequency below which 85% of energy)
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

        # Spectral flatness (geometric mean / arithmetic mean)
        log_spectrum = np.log(spectrum + 1e-10)
        geometric_mean = np.exp(np.mean(log_spectrum))
        arithmetic_mean = np.mean(spectrum)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)

        # Spectral contrast (simplified: ratio of peaks to valleys)
        # Using octave bands
        contrast = self._compute_spectral_contrast(spectrum, freqs)

        # Harmonic ratio (simplified estimate)
        harmonic_ratio = self._estimate_harmonic_ratio(audio, sample_rate)

        return SpectralQualityMetrics(
            spectral_centroid_hz=float(centroid),
            spectral_bandwidth_hz=float(bandwidth),
            spectral_rolloff_hz=float(rolloff),
            spectral_flatness=float(min(1.0, flatness)),
            spectral_contrast=float(contrast),
            harmonic_ratio=float(harmonic_ratio),
        )

    def _compute_spectral_contrast(
        self,
        spectrum: Any,
        freqs: Any,
    ) -> float:
        """Compute spectral contrast (peak-to-valley ratio)."""
        if len(spectrum) < 10:
            return 0.5

        # Simple approach: ratio of top 10% to bottom 10%
        sorted_spectrum = np.sort(spectrum)
        n = len(sorted_spectrum)
        top = np.mean(sorted_spectrum[int(0.9 * n):])
        bottom = np.mean(sorted_spectrum[:int(0.1 * n)]) + 1e-10

        contrast = np.log10(top / bottom + 1) / 3  # Normalize to ~0-1
        return float(min(1.0, max(0.0, contrast)))

    def _estimate_harmonic_ratio(
        self,
        audio: Any,
        sample_rate: int,
    ) -> float:
        """
        Estimate harmonic-to-noise ratio.

        Uses autocorrelation to detect periodicity (harmonicity).
        """
        if not NUMPY_AVAILABLE:
            return 0.8

        # Use a segment of audio
        segment_len = min(len(audio), sample_rate // 2)  # 500ms max
        segment = audio[:segment_len]

        # Autocorrelation
        corr = np.correlate(segment, segment, mode='full')
        corr = corr[len(corr) // 2:]  # Take positive lags

        # Find first peak after zero crossing (fundamental period)
        # Skip first few samples (too close to zero lag)
        min_lag = int(sample_rate / 500)  # Max 500 Hz fundamental
        max_lag = int(sample_rate / 50)   # Min 50 Hz fundamental

        if max_lag >= len(corr):
            max_lag = len(corr) - 1

        if min_lag >= max_lag:
            return 0.5

        search_region = corr[min_lag:max_lag]
        if len(search_region) == 0:
            return 0.5

        peak_idx = np.argmax(search_region)
        peak_value = search_region[peak_idx]

        # Harmonic ratio = peak correlation / zero-lag correlation
        harmonic_ratio = peak_value / (corr[0] + 1e-10)

        return float(min(1.0, max(0.0, harmonic_ratio)))

    # ========================================================================
    # NATURALNESS ESTIMATION
    # ========================================================================

    def estimate_naturalness(
        self,
        audio: Any,
        sample_rate: int = 24000,
    ) -> NaturalnessMetrics:
        """
        Estimate speech naturalness (heuristic-based).

        For production use, consider using a neural MOS predictor
        like NISQA or UTMOS. This implementation uses heuristics
        based on spectral and temporal characteristics.

        Args:
            audio: Numpy array of audio samples
            sample_rate: Sample rate in Hz

        Returns:
            NaturalnessMetrics instance
        """
        if not NUMPY_AVAILABLE or audio is None:
            return NaturalnessMetrics.from_mos(3.5)

        # Get spectral metrics for prosody estimation
        spectral = self.compute_spectral_metrics(audio, sample_rate)
        technical = self.compute_technical_metrics(audio, sample_rate)

        # Prosody score (based on spectral variation)
        prosody_score = self._estimate_prosody_naturalness(audio, sample_rate)

        # Articulation score (based on spectral clarity)
        articulation_score = self._estimate_articulation(spectral, technical)

        # Pacing variance (based on energy envelope)
        pacing_variance = self._estimate_pacing_variance(audio, sample_rate)

        # Breath naturalness (based on pause patterns)
        breath_naturalness = self._estimate_breath_naturalness(audio, sample_rate)

        # Combine into overall naturalness
        overall = (
            0.35 * prosody_score +
            0.25 * articulation_score +
            0.20 * (1.0 - min(1.0, pacing_variance * 5)) +
            0.20 * breath_naturalness
        )

        # Map overall score to MOS (1-5)
        mos = 1.0 + overall * 4.0

        return NaturalnessMetrics(
            mos_estimate=float(mos),
            prosody_score=float(prosody_score),
            articulation_score=float(articulation_score),
            pacing_variance=float(pacing_variance),
            breath_naturalness=float(breath_naturalness),
            overall_naturalness=float(overall),
        )

    def _estimate_prosody_naturalness(
        self,
        audio: Any,
        sample_rate: int,
    ) -> float:
        """Estimate prosody naturalness from pitch variation."""
        if not SCIPY_AVAILABLE:
            return 0.7

        # Compute short-time energy to find voiced regions
        window_size = int(sample_rate * 0.025)  # 25ms
        hop_size = int(sample_rate * 0.010)     # 10ms

        energies = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            energy = np.sum(window ** 2)
            energies.append(energy)

        energies = np.array(energies)
        if len(energies) < 10:
            return 0.7

        # Natural speech has moderate energy variation
        energy_std = np.std(energies) / (np.mean(energies) + 1e-10)

        # Too flat (0) or too variable (>1) is unnatural
        if 0.3 <= energy_std <= 0.8:
            prosody_score = 0.9
        elif energy_std < 0.3:
            prosody_score = 0.5 + energy_std
        else:
            prosody_score = max(0.4, 1.0 - (energy_std - 0.8) * 0.5)

        return prosody_score

    def _estimate_articulation(
        self,
        spectral: SpectralQualityMetrics,
        technical: TechnicalQualityMetrics,
    ) -> float:
        """Estimate articulation clarity from spectral characteristics."""
        # Good articulation: clear spectral peaks, good SNR
        snr_score = min(1.0, technical.snr_db / 30)
        flatness_score = 1.0 - spectral.spectral_flatness  # Less flat = more articulate
        harmonic_score = spectral.harmonic_ratio

        return 0.4 * snr_score + 0.3 * flatness_score + 0.3 * harmonic_score

    def _estimate_pacing_variance(
        self,
        audio: Any,
        sample_rate: int,
    ) -> float:
        """Estimate pacing variance (speaking rate consistency)."""
        if not NUMPY_AVAILABLE:
            return 0.1

        # Find energy peaks (syllable-like units)
        window_size = int(sample_rate * 0.05)  # 50ms
        hop_size = int(sample_rate * 0.01)     # 10ms

        energies = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            energies.append(np.sqrt(np.mean(window ** 2)))

        energies = np.array(energies)
        if len(energies) < 20:
            return 0.1

        # Find peaks in energy envelope
        threshold = np.mean(energies) + 0.5 * np.std(energies)
        peaks = np.where(energies > threshold)[0]

        if len(peaks) < 2:
            return 0.1

        # Compute inter-peak intervals
        intervals = np.diff(peaks)
        if len(intervals) < 2:
            return 0.1

        # Variance of intervals (normalized)
        variance = np.std(intervals) / (np.mean(intervals) + 1e-10)

        return float(min(1.0, variance))

    def _estimate_breath_naturalness(
        self,
        audio: Any,
        sample_rate: int,
    ) -> float:
        """Estimate breath/pause naturalness."""
        if not NUMPY_AVAILABLE:
            return 0.7

        # Find silent regions (potential pauses)
        window_size = int(sample_rate * 0.02)  # 20ms
        hop_size = window_size // 2

        is_silence = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            is_silence.append(rms < self._silence_threshold_linear)

        if not is_silence:
            return 0.7

        is_silence = np.array(is_silence)

        # Find pause durations
        pause_starts = np.where(np.diff(is_silence.astype(int)) == 1)[0]
        pause_ends = np.where(np.diff(is_silence.astype(int)) == -1)[0]

        if len(pause_starts) == 0 or len(pause_ends) == 0:
            return 0.7

        # Compute pause durations in ms
        min_len = min(len(pause_starts), len(pause_ends))
        pause_durations = (pause_ends[:min_len] - pause_starts[:min_len]) * hop_size / sample_rate * 1000

        if len(pause_durations) == 0:
            return 0.7

        # Natural speech has pauses mostly 100-500ms
        natural_pauses = np.sum((pause_durations >= 100) & (pause_durations <= 500))
        naturalness = natural_pauses / len(pause_durations)

        return float(min(1.0, naturalness + 0.3))  # Boost a bit

    # ========================================================================
    # ARTIFACT DETECTION (DETAILED)
    # ========================================================================

    def detect_artifacts(
        self,
        audio: Any,
        sample_rate: int = 24000,
    ) -> Tuple[float, List[Tuple[float, float, str]]]:
        """
        Detect audio artifacts with locations.

        Args:
            audio: Numpy array of audio samples
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (artifact_score, [(start_s, end_s, type), ...])
        """
        if not NUMPY_AVAILABLE or audio is None:
            return 0.1, []

        artifacts = []

        # 1. Detect clicks/pops (sudden amplitude changes)
        diff = np.abs(np.diff(audio))
        threshold = np.mean(diff) + 5 * np.std(diff)
        click_indices = np.where(diff > threshold)[0]

        for idx in click_indices[:10]:  # Limit to first 10
            time_s = idx / sample_rate
            artifacts.append((time_s, time_s + 0.001, "click"))

        # 2. Detect clipping
        clip_indices = np.where(np.abs(audio) > self.clipping_threshold)[0]
        if len(clip_indices) > 0:
            # Group consecutive clipped samples
            groups = np.split(clip_indices, np.where(np.diff(clip_indices) > 1)[0] + 1)
            for group in groups[:5]:  # Limit to first 5
                if len(group) > 0:
                    start_s = group[0] / sample_rate
                    end_s = group[-1] / sample_rate
                    artifacts.append((start_s, end_s, "clipping"))

        # 3. Detect silence gaps (potential glitches)
        window_size = int(sample_rate * 0.01)  # 10ms
        for i in range(0, len(audio) - window_size * 3, window_size):
            # Check for sudden silence surrounded by audio
            before = np.sqrt(np.mean(audio[i:i+window_size] ** 2))
            middle = np.sqrt(np.mean(audio[i+window_size:i+window_size*2] ** 2))
            after = np.sqrt(np.mean(audio[i+window_size*2:i+window_size*3] ** 2))

            if before > 0.1 and after > 0.1 and middle < 0.01:
                time_s = (i + window_size) / sample_rate
                artifacts.append((time_s, time_s + window_size / sample_rate, "dropout"))

        # Compute overall artifact score
        artifact_count = len(artifacts)
        audio_duration = len(audio) / sample_rate
        artifact_score = min(1.0, artifact_count / (audio_duration * 10 + 1))

        return artifact_score, artifacts


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  AXIOM VOX Audio Analyzer Demo")
    print("=" * 70)

    analyzer = VoxAudioAnalyzer()

    if NUMPY_AVAILABLE:
        # Generate a test signal
        sample_rate = 24000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Simulate speech-like signal (sum of harmonics)
        audio = (
            0.3 * np.sin(2 * np.pi * 150 * t) +  # Fundamental
            0.2 * np.sin(2 * np.pi * 300 * t) +  # 2nd harmonic
            0.1 * np.sin(2 * np.pi * 450 * t) +  # 3rd harmonic
            0.05 * np.random.randn(len(t))       # Noise
        ).astype(np.float32)

        print("\nTest signal: 1s speech-like waveform")
        print("-" * 70)

        # Technical metrics
        tech = analyzer.compute_technical_metrics(audio, sample_rate)
        print(f"\nTechnical Quality:")
        print(f"  SNR: {tech.snr_db:.1f} dB")
        print(f"  Peak: {tech.peak_amplitude:.3f}")
        print(f"  RMS: {tech.rms_level_db:.1f} dB")
        print(f"  Dynamic Range: {tech.dynamic_range_db:.1f} dB")
        print(f"  Artifact Score: {tech.artifact_score:.3f}")
        print(f"  Quality Score: {tech.get_quality_score():.2f}")

        # Spectral metrics
        if SCIPY_AVAILABLE:
            spectral = analyzer.compute_spectral_metrics(audio, sample_rate)
            print(f"\nSpectral Quality:")
            print(f"  Centroid: {spectral.spectral_centroid_hz:.0f} Hz")
            print(f"  Bandwidth: {spectral.spectral_bandwidth_hz:.0f} Hz")
            print(f"  Rolloff: {spectral.spectral_rolloff_hz:.0f} Hz")
            print(f"  Flatness: {spectral.spectral_flatness:.3f}")
            print(f"  Harmonic Ratio: {spectral.harmonic_ratio:.3f}")
            print(f"  Quality Score: {spectral.get_quality_score():.2f}")

        # Naturalness metrics
        naturalness = analyzer.estimate_naturalness(audio, sample_rate)
        print(f"\nNaturalness:")
        print(f"  MOS Estimate: {naturalness.mos_estimate:.2f}")
        print(f"  Prosody: {naturalness.prosody_score:.2f}")
        print(f"  Articulation: {naturalness.articulation_score:.2f}")
        print(f"  Overall: {naturalness.overall_naturalness:.2f}")

        # Artifact detection
        score, artifacts = analyzer.detect_artifacts(audio, sample_rate)
        print(f"\nArtifact Detection:")
        print(f"  Score: {score:.3f}")
        print(f"  Artifacts found: {len(artifacts)}")

    else:
        print("\nnumpy not available - showing default values")
        tech = analyzer.compute_technical_metrics(None, 24000)
        print(f"Default Technical Quality Score: {tech.get_quality_score():.2f}")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
