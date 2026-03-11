"""
VØX Voice Genome - Feature Extractor
------------------------------------

Low-level acoustic feature extraction from voice audio.

Extracts the raw measurements used by biometric, psychometric,
and sociometric analyzers.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np

from .models import AcousticFeatures

logger = logging.getLogger(__name__)

# Try to import optional audio libraries
try:
    import scipy.signal as signal
    from scipy.fft import fft, fftfreq
    from scipy.stats import kurtosis, skew
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - feature extraction will be limited")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.debug("librosa not available - using basic extraction")


@dataclass
class ExtractorConfig:
    """
    Configuration for feature extractor.

    Attributes:
        sample_rate: Target sample rate
        frame_length_ms: Frame length in milliseconds
        hop_length_ms: Hop length in milliseconds
        n_mfcc: Number of MFCCs to extract
        f0_min: Minimum fundamental frequency
        f0_max: Maximum fundamental frequency
    """
    sample_rate: int = 22050
    frame_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    n_mfcc: int = 13
    f0_min: float = 50.0
    f0_max: float = 500.0


class VoiceFeatureExtractor:
    """
    Extracts low-level acoustic features from voice audio.

    Features extracted:
        - Fundamental frequency (F0) statistics
        - Voice quality (jitter, shimmer, HNR)
        - Formants (F1-F4)
        - Temporal features (rate, pauses)
        - Intensity/energy features
        - Spectral features (centroid, MFCCs)
    """

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
    ):
        """
        Initialize feature extractor.

        Args:
            config: Extractor configuration
        """
        self.config = config or ExtractorConfig()

        # Compute frame/hop in samples
        self.frame_length = int(self.config.frame_length_ms * self.config.sample_rate / 1000)
        self.hop_length = int(self.config.hop_length_ms * self.config.sample_rate / 1000)

    def extract(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: Optional[int] = None,
    ) -> AcousticFeatures:
        """
        Extract all acoustic features from audio.

        Args:
            audio: Audio data or path to audio file
            sample_rate: Sample rate (required if audio is array)

        Returns:
            Extracted acoustic features
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio(str(audio))

        if sample_rate is None:
            sample_rate = self.config.sample_rate

        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        features = AcousticFeatures()

        # Extract F0 (fundamental frequency)
        f0_features = self._extract_f0(audio, sample_rate)
        features.f0_mean = f0_features["mean"]
        features.f0_std = f0_features["std"]
        features.f0_min = f0_features["min"]
        features.f0_max = f0_features["max"]
        features.f0_range = f0_features["range"]

        # Extract voice quality
        quality = self._extract_voice_quality(audio, sample_rate)
        features.jitter_percent = quality["jitter"]
        features.shimmer_percent = quality["shimmer"]
        features.hnr_db = quality["hnr"]
        features.nhr = quality["nhr"]

        # Extract formants
        formants = self._extract_formants(audio, sample_rate)
        features.f1_mean = formants["f1"]
        features.f2_mean = formants["f2"]
        features.f3_mean = formants["f3"]
        features.f4_mean = formants["f4"]
        features.formant_dispersion = formants["dispersion"]

        # Extract temporal features
        temporal = self._extract_temporal(audio, sample_rate)
        features.speaking_rate = temporal["speaking_rate"]
        features.pause_ratio = temporal["pause_ratio"]
        features.mean_pause_duration = temporal["mean_pause"]

        # Extract intensity features
        intensity = self._extract_intensity(audio, sample_rate)
        features.intensity_mean = intensity["mean"]
        features.intensity_std = intensity["std"]
        features.intensity_range = intensity["range"]

        # Extract spectral features
        spectral = self._extract_spectral(audio, sample_rate)
        features.spectral_centroid = spectral["centroid"]
        features.spectral_spread = spectral["spread"]
        features.spectral_slope = spectral["slope"]
        features.spectral_flux = spectral["flux"]
        features.mfcc_means = spectral["mfcc_means"]

        return features

    def extract_from_file(self, path: str) -> AcousticFeatures:
        """Extract features from audio file."""
        return self.extract(path)

    def _load_audio(
        self,
        path: str,
    ) -> Tuple[np.ndarray, int]:
        """Load audio from file."""
        if LIBROSA_AVAILABLE:
            audio, sr = librosa.load(path, sr=self.config.sample_rate)
            return audio, sr

        try:
            import soundfile as sf
            audio, sr = sf.read(path)
            return audio, sr
        except ImportError:
            pass

        logger.warning(f"Cannot load {path} - no audio library available")
        return np.zeros(self.config.sample_rate * 10), self.config.sample_rate

    def _extract_f0(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """Extract fundamental frequency features."""
        if LIBROSA_AVAILABLE:
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    audio,
                    fmin=self.config.f0_min,
                    fmax=self.config.f0_max,
                    sr=sample_rate,
                )

                # Filter to voiced frames
                f0_voiced = f0[~np.isnan(f0)]

                if len(f0_voiced) > 0:
                    return {
                        "mean": float(np.mean(f0_voiced)),
                        "std": float(np.std(f0_voiced)),
                        "min": float(np.min(f0_voiced)),
                        "max": float(np.max(f0_voiced)),
                        "range": float(np.max(f0_voiced) - np.min(f0_voiced)),
                    }
            except Exception as e:
                logger.debug(f"F0 extraction failed: {e}")

        # Fallback: autocorrelation-based pitch
        return self._autocorr_pitch(audio, sample_rate)

    def _autocorr_pitch(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """Autocorrelation-based pitch estimation."""
        if not SCIPY_AVAILABLE:
            return {"mean": 150.0, "std": 30.0, "min": 100.0, "max": 200.0, "range": 100.0}

        try:
            # Simple autocorrelation pitch detection
            frame_size = int(0.05 * sample_rate)  # 50ms frames
            hop = frame_size // 2
            n_frames = (len(audio) - frame_size) // hop

            f0_values = []

            min_lag = int(sample_rate / self.config.f0_max)
            max_lag = int(sample_rate / self.config.f0_min)

            for i in range(n_frames):
                frame = audio[i * hop:i * hop + frame_size]
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]

                if max_lag < len(autocorr):
                    search = autocorr[min_lag:max_lag]
                    if len(search) > 0:
                        peak = np.argmax(search) + min_lag
                        if autocorr[peak] > 0.3 * autocorr[0]:  # Voiced threshold
                            f0 = sample_rate / peak
                            if self.config.f0_min <= f0 <= self.config.f0_max:
                                f0_values.append(f0)

            if f0_values:
                return {
                    "mean": float(np.mean(f0_values)),
                    "std": float(np.std(f0_values)),
                    "min": float(np.min(f0_values)),
                    "max": float(np.max(f0_values)),
                    "range": float(np.max(f0_values) - np.min(f0_values)),
                }

        except Exception as e:
            logger.debug(f"Autocorr pitch failed: {e}")

        return {"mean": 150.0, "std": 30.0, "min": 100.0, "max": 200.0, "range": 100.0}

    def _extract_voice_quality(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """Extract voice quality measures (jitter, shimmer, HNR)."""
        result = {"jitter": 0.0, "shimmer": 0.0, "hnr": 20.0, "nhr": 0.05}

        if not SCIPY_AVAILABLE:
            return result

        try:
            # Find pitch periods
            frame_size = int(0.03 * sample_rate)
            periods = []
            amplitudes = []

            for i in range(0, len(audio) - frame_size, frame_size // 2):
                frame = audio[i:i + frame_size]

                # Autocorrelation to find period
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]

                min_lag = int(sample_rate / 400)  # 400 Hz max
                max_lag = int(sample_rate / 75)   # 75 Hz min

                if max_lag < len(autocorr):
                    search = autocorr[min_lag:max_lag]
                    if len(search) > 0 and np.max(search) > 0.3 * autocorr[0]:
                        peak = np.argmax(search) + min_lag
                        periods.append(peak / sample_rate)
                        amplitudes.append(np.max(np.abs(frame)))

            if len(periods) > 2:
                # Jitter: period-to-period variation
                period_diffs = np.abs(np.diff(periods))
                jitter = np.mean(period_diffs) / np.mean(periods) * 100
                result["jitter"] = min(5.0, jitter)

                # Shimmer: amplitude variation
                amp_diffs = np.abs(np.diff(amplitudes))
                shimmer = np.mean(amp_diffs) / np.mean(amplitudes) * 100
                result["shimmer"] = min(10.0, shimmer)

            # HNR estimation
            hnr = self._estimate_hnr(audio, sample_rate)
            result["hnr"] = hnr
            result["nhr"] = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 0.1

        except Exception as e:
            logger.debug(f"Voice quality extraction failed: {e}")

        return result

    def _estimate_hnr(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate Harmonics-to-Noise Ratio."""
        try:
            frame_size = int(0.05 * sample_rate)
            hnr_values = []

            for i in range(0, len(audio) - frame_size, frame_size):
                frame = audio[i:i + frame_size]

                # Autocorrelation
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr)//2:]

                if len(autocorr) > 1:
                    # Find first significant peak
                    r0 = autocorr[0]
                    peaks = []
                    for j in range(20, len(autocorr) - 1):
                        if autocorr[j] > autocorr[j-1] and autocorr[j] > autocorr[j+1]:
                            if autocorr[j] > 0.2 * r0:
                                peaks.append(autocorr[j])
                                break

                    if peaks:
                        r_max = peaks[0]
                        if r0 > r_max and r_max > 0:
                            hnr = 10 * np.log10(r_max / (r0 - r_max))
                            hnr_values.append(hnr)

            if hnr_values:
                return float(np.median(hnr_values))

        except Exception:
            pass

        return 15.0  # Default

    def _extract_formants(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """Extract formant frequencies."""
        result = {"f1": 500.0, "f2": 1500.0, "f3": 2500.0, "f4": 3500.0, "dispersion": 1000.0}

        if not SCIPY_AVAILABLE:
            return result

        try:
            # LPC-based formant estimation
            from scipy.signal import lfilter

            # Pre-emphasis
            pre_emphasis = 0.97
            emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

            # Frame the signal
            frame_size = int(0.025 * sample_rate)
            frame = emphasized[:frame_size]

            # Apply window
            windowed = frame * np.hamming(len(frame))

            # LPC using autocorrelation method
            order = int(sample_rate / 1000) + 4  # ~2 + sr/1000

            # Autocorrelation
            r = np.correlate(windowed, windowed, mode='full')
            r = r[len(r)//2:len(r)//2 + order + 1]

            # Levinson-Durbin
            a = np.zeros(order + 1)
            a[0] = 1.0
            e = r[0]

            for i in range(1, order + 1):
                lambda_val = 0
                for j in range(i):
                    lambda_val -= a[j] * r[i - j]
                lambda_val /= e

                a_new = a.copy()
                for j in range(i):
                    a_new[j] = a[j] + lambda_val * a[i - 1 - j]
                a_new[i] = lambda_val
                a = a_new

                e *= (1 - lambda_val ** 2)

            # Find roots
            roots = np.roots(a)

            # Get formants from roots
            formants = []
            for root in roots:
                if np.imag(root) > 0:
                    freq = np.abs(np.arctan2(np.imag(root), np.real(root))) * sample_rate / (2 * np.pi)
                    if 90 < freq < 5000:
                        formants.append(freq)

            formants = sorted(formants)

            if len(formants) >= 1:
                result["f1"] = formants[0]
            if len(formants) >= 2:
                result["f2"] = formants[1]
            if len(formants) >= 3:
                result["f3"] = formants[2]
            if len(formants) >= 4:
                result["f4"] = formants[3]

            # Formant dispersion
            if len(formants) >= 3:
                result["dispersion"] = (formants[2] - formants[0]) / 2

        except Exception as e:
            logger.debug(f"Formant extraction failed: {e}")

        return result

    def _extract_temporal(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """Extract temporal features (speaking rate, pauses)."""
        result = {"speaking_rate": 4.0, "pause_ratio": 0.3, "mean_pause": 0.2}

        try:
            # Energy envelope
            frame_size = int(0.02 * sample_rate)
            hop = frame_size // 2
            n_frames = (len(audio) - frame_size) // hop

            energy = np.zeros(n_frames)
            for i in range(n_frames):
                frame = audio[i * hop:i * hop + frame_size]
                energy[i] = np.sum(frame ** 2)

            # Normalize
            if np.max(energy) > 0:
                energy = energy / np.max(energy)

            # Detect voiced/unvoiced
            threshold = 0.1
            voiced = energy > threshold

            # Count syllables (rough: energy peaks)
            peaks = 0
            for i in range(1, len(energy) - 1):
                if energy[i] > energy[i-1] and energy[i] > energy[i+1]:
                    if energy[i] > threshold * 2:
                        peaks += 1

            duration = len(audio) / sample_rate
            if duration > 0:
                result["speaking_rate"] = peaks / duration

            # Pause analysis
            pause_frames = np.sum(~voiced)
            result["pause_ratio"] = pause_frames / len(energy) if len(energy) > 0 else 0.3

            # Find pause durations
            pauses = []
            in_pause = False
            pause_start = 0
            for i, v in enumerate(voiced):
                if not v and not in_pause:
                    in_pause = True
                    pause_start = i
                elif v and in_pause:
                    pause_duration = (i - pause_start) * hop / sample_rate
                    if pause_duration > 0.1:  # Only count pauses > 100ms
                        pauses.append(pause_duration)
                    in_pause = False

            if pauses:
                result["mean_pause"] = np.mean(pauses)

        except Exception as e:
            logger.debug(f"Temporal extraction failed: {e}")

        return result

    def _extract_intensity(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, float]:
        """Extract intensity/energy features."""
        result = {"mean": -20.0, "std": 5.0, "range": 20.0}

        try:
            # Frame-based RMS
            frame_size = int(0.02 * sample_rate)
            hop = frame_size // 2
            n_frames = (len(audio) - frame_size) // hop

            rms = []
            for i in range(n_frames):
                frame = audio[i * hop:i * hop + frame_size]
                rms_val = np.sqrt(np.mean(frame ** 2))
                if rms_val > 1e-10:
                    rms.append(20 * np.log10(rms_val))

            if rms:
                result["mean"] = float(np.mean(rms))
                result["std"] = float(np.std(rms))
                result["range"] = float(np.max(rms) - np.min(rms))

        except Exception as e:
            logger.debug(f"Intensity extraction failed: {e}")

        return result

    def _extract_spectral(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, Any]:
        """Extract spectral features."""
        result = {
            "centroid": 2000.0,
            "spread": 1000.0,
            "slope": -0.5,
            "flux": 0.1,
            "mfcc_means": [0.0] * self.config.n_mfcc,
        }

        if LIBROSA_AVAILABLE:
            try:
                # Spectral centroid
                centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
                result["centroid"] = float(np.mean(centroid))

                # Spectral bandwidth (spread)
                bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
                result["spread"] = float(np.mean(bandwidth))

                # Spectral rolloff (related to slope)
                rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
                result["slope"] = float(np.mean(rolloff)) / sample_rate - 0.5

                # Spectral flux
                onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
                result["flux"] = float(np.mean(onset_env))

                # MFCCs
                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.config.n_mfcc)
                result["mfcc_means"] = [float(np.mean(mfccs[i])) for i in range(self.config.n_mfcc)]

            except Exception as e:
                logger.debug(f"Spectral extraction failed: {e}")

        elif SCIPY_AVAILABLE:
            # Basic spectral features
            try:
                n = len(audio)
                spectrum = np.abs(fft(audio))[:n//2]
                freqs = fftfreq(n, 1/sample_rate)[:n//2]

                # Centroid
                if np.sum(spectrum) > 0:
                    result["centroid"] = float(np.sum(freqs * spectrum) / np.sum(spectrum))

                # Spread
                centroid = result["centroid"]
                result["spread"] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum)))

            except Exception as e:
                logger.debug(f"Basic spectral extraction failed: {e}")

        return result


def extract_voice_features(
    audio_path: str,
    config: Optional[ExtractorConfig] = None,
) -> AcousticFeatures:
    """
    Extract acoustic features from voice audio file.

    Args:
        audio_path: Path to audio file
        config: Optional extractor configuration

    Returns:
        Extracted acoustic features
    """
    extractor = VoiceFeatureExtractor(config)
    return extractor.extract_from_file(audio_path)
