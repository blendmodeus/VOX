"""
VØX Resonance - Audio Analyzer
------------------------------

Extracts audio features relevant to psychological resonance.

Features:
    - Tempo/BPM detection
    - Key detection (major/minor)
    - Spectral analysis (bass, mid, treble energy)
    - Dynamic range analysis
    - Rhythm regularity
    - Harmonic complexity
    - Dissonance measurement
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np

from .models import AudioFeatures, MusicalKey

logger = logging.getLogger(__name__)

# Try to import optional audio libraries
try:
    import scipy.signal as signal
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - audio analysis will be limited")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.debug("librosa not available - using basic analysis")


@dataclass
class AudioAnalyzerConfig:
    """
    Configuration for audio analyzer.

    Attributes:
        sample_rate: Target sample rate for analysis
        hop_length: Hop length for STFT
        n_fft: FFT window size
        bass_range: Frequency range for bass (Hz)
        mid_range: Frequency range for mids (Hz)
        treble_range: Frequency range for treble (Hz)
    """
    sample_rate: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    bass_range: Tuple[int, int] = (20, 250)
    mid_range: Tuple[int, int] = (250, 4000)
    treble_range: Tuple[int, int] = (4000, 20000)


class AudioAnalyzer:
    """
    Analyzer for extracting resonance-relevant audio features.

    Works with or without librosa - uses basic numpy/scipy fallbacks.
    """

    def __init__(
        self,
        config: Optional[AudioAnalyzerConfig] = None,
    ):
        """
        Initialize audio analyzer.

        Args:
            config: Analyzer configuration
        """
        self.config = config or AudioAnalyzerConfig()

    def analyze(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: Optional[int] = None,
    ) -> AudioFeatures:
        """
        Analyze audio and extract features.

        Args:
            audio: Audio data as numpy array or path to audio file
            sample_rate: Sample rate (required if audio is array)

        Returns:
            Extracted audio features
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            audio, sample_rate = self._load_audio(str(audio))

        if sample_rate is None:
            sample_rate = self.config.sample_rate

        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Extract all features
        features = AudioFeatures()

        # Basic features
        features.duration_seconds = len(audio) / sample_rate

        # Tempo
        features.tempo_bpm = self._estimate_tempo(audio, sample_rate)

        # Key detection
        features.key, features.is_major = self._detect_key(audio, sample_rate)

        # Loudness
        features.loudness_db = self._estimate_loudness(audio)

        # Dynamic range
        features.dynamic_range_db = self._estimate_dynamic_range(audio, sample_rate)

        # Spectral energy distribution
        bass, mid, treble = self._analyze_spectral_energy(audio, sample_rate)
        features.bass_energy = bass
        features.mid_energy = mid
        features.treble_energy = treble

        # Spectral centroid (brightness)
        features.spectral_centroid = self._estimate_spectral_centroid(audio, sample_rate)

        # Harmonic complexity
        features.spectral_complexity = self._estimate_harmonic_complexity(audio, sample_rate)

        # Dissonance
        features.dissonance_score = self._estimate_dissonance(audio, sample_rate)

        # Rhythm regularity
        features.rhythm_regularity = self._estimate_rhythm_regularity(audio, sample_rate)

        # Vocal presence (rough estimate)
        features.vocal_presence = self._estimate_vocal_presence(audio, sample_rate)

        return features

    def analyze_file(self, path: str) -> AudioFeatures:
        """
        Analyze audio file.

        Args:
            path: Path to audio file

        Returns:
            Extracted audio features
        """
        return self.analyze(path)

    def _load_audio(
        self,
        path: str,
    ) -> Tuple[np.ndarray, int]:
        """Load audio from file."""
        if LIBROSA_AVAILABLE:
            audio, sr = librosa.load(path, sr=self.config.sample_rate)
            return audio, sr

        # Fallback: try soundfile
        try:
            import soundfile as sf
            audio, sr = sf.read(path)
            return audio, sr
        except ImportError:
            pass

        # Last resort: return placeholder
        logger.warning(f"Cannot load {path} - no audio library available")
        return np.zeros(self.config.sample_rate * 180), self.config.sample_rate

    def _estimate_tempo(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate tempo in BPM."""
        if LIBROSA_AVAILABLE:
            try:
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
                return float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
            except Exception:
                pass

        # Fallback: onset-based estimation
        if SCIPY_AVAILABLE:
            # Simple onset detection using energy envelope
            frame_length = int(0.02 * sample_rate)  # 20ms frames
            hop = frame_length // 2

            # Compute energy envelope
            n_frames = (len(audio) - frame_length) // hop + 1
            energy = np.zeros(n_frames)

            for i in range(n_frames):
                start = i * hop
                frame = audio[start:start + frame_length]
                energy[i] = np.sum(frame ** 2)

            # Find peaks (onsets)
            energy_diff = np.diff(energy)
            energy_diff = np.maximum(energy_diff, 0)

            # Autocorrelation to find tempo
            if len(energy_diff) > 100:
                autocorr = np.correlate(energy_diff, energy_diff, mode='full')
                autocorr = autocorr[len(autocorr)//2:]

                # Find first major peak after 0.25 seconds
                min_lag = int(0.25 * sample_rate / hop)
                max_lag = int(2.0 * sample_rate / hop)

                if max_lag < len(autocorr):
                    search_region = autocorr[min_lag:max_lag]
                    if len(search_region) > 0:
                        peak_idx = np.argmax(search_region) + min_lag
                        period_seconds = peak_idx * hop / sample_rate
                        if period_seconds > 0:
                            return 60.0 / period_seconds

        return 120.0  # Default tempo

    def _detect_key(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[MusicalKey, bool]:
        """Detect musical key."""
        if LIBROSA_AVAILABLE:
            try:
                # Compute chromagram
                chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
                chroma_mean = np.mean(chroma, axis=1)

                # Key profiles (Krumhansl-Schmuckler)
                major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                         2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
                minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                         2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

                # Correlate with all keys
                keys = ['C', 'C#', 'D', 'D#', 'E', 'F',
                       'F#', 'G', 'G#', 'A', 'A#', 'B']
                best_corr = -1
                best_key = 'C'
                is_major = True

                for i in range(12):
                    rotated = np.roll(chroma_mean, -i)

                    major_corr = np.corrcoef(rotated, major_profile)[0, 1]
                    minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]

                    if major_corr > best_corr:
                        best_corr = major_corr
                        best_key = keys[i]
                        is_major = True

                    if minor_corr > best_corr:
                        best_corr = minor_corr
                        best_key = keys[i]
                        is_major = False

                # Map to enum
                key_str = f"{best_key} {'major' if is_major else 'minor'}"
                for k in MusicalKey:
                    if k.value.lower().replace('#', '#') == key_str.lower():
                        return k, is_major

                return MusicalKey.UNKNOWN, is_major

            except Exception as e:
                logger.debug(f"Key detection failed: {e}")

        return MusicalKey.UNKNOWN, True

    def _estimate_loudness(self, audio: np.ndarray) -> float:
        """Estimate average loudness in dB."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            db = 20 * np.log10(rms)
            return max(-60, min(0, db))
        return -60.0

    def _estimate_dynamic_range(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate dynamic range in dB."""
        # Calculate RMS in windows
        window_size = int(0.1 * sample_rate)  # 100ms windows
        n_windows = len(audio) // window_size

        if n_windows < 2:
            return 10.0

        rms_values = []
        for i in range(n_windows):
            start = i * window_size
            window = audio[start:start + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            if rms > 1e-10:
                rms_values.append(rms)

        if len(rms_values) < 2:
            return 10.0

        rms_values = np.array(rms_values)
        loud = np.percentile(rms_values, 95)
        quiet = np.percentile(rms_values, 10)

        if quiet > 0 and loud > quiet:
            return 20 * np.log10(loud / quiet)

        return 10.0

    def _analyze_spectral_energy(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Tuple[float, float, float]:
        """Analyze energy distribution across frequency bands."""
        if not SCIPY_AVAILABLE:
            return 0.33, 0.34, 0.33

        try:
            # Compute spectrum
            n = len(audio)
            spectrum = np.abs(fft(audio))[:n//2]
            freqs = fftfreq(n, 1/sample_rate)[:n//2]

            # Calculate energy in each band
            bass_mask = (freqs >= self.config.bass_range[0]) & \
                       (freqs <= self.config.bass_range[1])
            mid_mask = (freqs >= self.config.mid_range[0]) & \
                      (freqs <= self.config.mid_range[1])
            treble_mask = (freqs >= self.config.treble_range[0]) & \
                         (freqs <= self.config.treble_range[1])

            bass_energy = np.sum(spectrum[bass_mask] ** 2)
            mid_energy = np.sum(spectrum[mid_mask] ** 2)
            treble_energy = np.sum(spectrum[treble_mask] ** 2)

            total = bass_energy + mid_energy + treble_energy
            if total > 0:
                return (
                    bass_energy / total,
                    mid_energy / total,
                    treble_energy / total,
                )

        except Exception as e:
            logger.debug(f"Spectral analysis failed: {e}")

        return 0.33, 0.34, 0.33

    def _estimate_spectral_centroid(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate spectral centroid (brightness)."""
        if LIBROSA_AVAILABLE:
            try:
                centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
                return float(np.mean(centroid))
            except Exception:
                pass

        if SCIPY_AVAILABLE:
            try:
                n = len(audio)
                spectrum = np.abs(fft(audio))[:n//2]
                freqs = fftfreq(n, 1/sample_rate)[:n//2]

                if np.sum(spectrum) > 0:
                    return float(np.sum(freqs * spectrum) / np.sum(spectrum))
            except Exception:
                pass

        return 2000.0

    def _estimate_harmonic_complexity(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate harmonic complexity (0-1)."""
        if LIBROSA_AVAILABLE:
            try:
                # Use spectral flatness as inverse of harmonicity
                flatness = librosa.feature.spectral_flatness(y=audio)
                # Higher flatness = more noise-like = less harmonic
                # Invert and scale to 0-1 for complexity
                complexity = 1 - np.mean(flatness)
                return float(np.clip(complexity, 0, 1))
            except Exception:
                pass

        return 0.5

    def _estimate_dissonance(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate average dissonance (0-1)."""
        if LIBROSA_AVAILABLE:
            try:
                # Compute chroma and look for semitone clashes
                chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)

                # Dissonant intervals: minor 2nd, major 7th, tritone
                dissonance_sum = 0
                for i in range(chroma.shape[1]):
                    frame = chroma[:, i]
                    # Check for simultaneous semitones
                    for j in range(12):
                        # Minor 2nd
                        dissonance_sum += frame[j] * frame[(j+1) % 12] * 1.0
                        # Tritone
                        dissonance_sum += frame[j] * frame[(j+6) % 12] * 0.7

                avg_dissonance = dissonance_sum / chroma.shape[1] / 12
                return float(np.clip(avg_dissonance, 0, 1))

            except Exception:
                pass

        return 0.2

    def _estimate_rhythm_regularity(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate how regular/steady the rhythm is (0-1)."""
        if LIBROSA_AVAILABLE:
            try:
                # Get beat times
                tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
                if len(beats) < 3:
                    return 0.5

                # Calculate inter-beat intervals
                beat_times = librosa.frames_to_time(beats, sr=sample_rate)
                intervals = np.diff(beat_times)

                if len(intervals) > 0:
                    # Regularity = inverse of coefficient of variation
                    cv = np.std(intervals) / np.mean(intervals)
                    regularity = 1 / (1 + cv)
                    return float(np.clip(regularity, 0, 1))

            except Exception:
                pass

        return 0.8

    def _estimate_vocal_presence(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate vocal presence (rough heuristic)."""
        # Vocals typically have strong energy in 300-3000 Hz range
        # with characteristic formant patterns

        if SCIPY_AVAILABLE:
            try:
                n = len(audio)
                spectrum = np.abs(fft(audio))[:n//2]
                freqs = fftfreq(n, 1/sample_rate)[:n//2]

                # Vocal range
                vocal_mask = (freqs >= 300) & (freqs <= 3000)
                full_mask = (freqs >= 20) & (freqs <= 8000)

                vocal_energy = np.sum(spectrum[vocal_mask] ** 2)
                total_energy = np.sum(spectrum[full_mask] ** 2)

                if total_energy > 0:
                    ratio = vocal_energy / total_energy
                    # Normalize to 0-1 (vocals typically 40-70% of this range)
                    return float(np.clip(ratio * 1.5, 0, 1))

            except Exception:
                pass

        return 0.5


def analyze_audio(
    audio_path: str,
    config: Optional[AudioAnalyzerConfig] = None,
) -> AudioFeatures:
    """
    Analyze audio file for resonance features.

    Args:
        audio_path: Path to audio file
        config: Optional analyzer config

    Returns:
        Extracted audio features
    """
    analyzer = AudioAnalyzer(config)
    return analyzer.analyze_file(audio_path)
