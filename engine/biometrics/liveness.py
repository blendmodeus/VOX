"""
Voice Liveness Detection
------------------------

Anti-spoofing detection for voice biometric verification.

Detects:
    - Replay attacks (recorded speech playback)
    - Deepfakes (TTS-generated speech)
    - Channel anomalies (unnatural recording artifacts)

AXIØM Phase 5: Resonance - "finding signature frequency"
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

import numpy as np

try:
    from scipy import signal
    from scipy.fft import rfft, fft
    from scipy.stats import kurtosis, skew
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .models import LivenessResult, LivenessStatus

logger = logging.getLogger(__name__)


class LivenessCheck(str, Enum):
    """Types of liveness checks."""
    REPLAY = "replay"
    DEEPFAKE = "deepfake"
    BREATH = "breath"
    CHANNEL = "channel"
    PROSODY = "prosody"


@dataclass
class CheckResult:
    """Result of a single liveness check."""
    check_type: LivenessCheck
    score: float  # 0-1, higher = more likely real
    passed: bool
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


class ReplayDetector:
    """
    Detect replay attacks (recorded speech playback).

    Techniques:
        - Spectral analysis for compression artifacts
        - Channel noise pattern analysis
        - Frequency response anomalies
        - Pop filter / room characteristics
    """

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> CheckResult:
        """
        Detect if audio is a replay attack.

        Args:
            audio: Audio samples
            sample_rate: Sample rate

        Returns:
            CheckResult with replay detection score
        """
        if not SCIPY_AVAILABLE:
            return CheckResult(
                check_type=LivenessCheck.REPLAY,
                score=0.5,
                passed=True,
                threshold=self.threshold,
                message="scipy not available, skipping replay detection",
            )

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        scores = []
        details = {}

        # 1. Check for compression artifacts (MP3, AAC)
        compression_score = self._check_compression_artifacts(audio, sample_rate)
        scores.append(compression_score)
        details["compression_score"] = compression_score

        # 2. Check for speaker/mic channel patterns
        channel_score = self._check_channel_patterns(audio, sample_rate)
        scores.append(channel_score)
        details["channel_score"] = channel_score

        # 3. Check for unnaturally clean signal (too perfect)
        naturalness_score = self._check_signal_naturalness(audio, sample_rate)
        scores.append(naturalness_score)
        details["naturalness_score"] = naturalness_score

        # 4. Check frequency response for replay artifacts
        frequency_score = self._check_frequency_response(audio, sample_rate)
        scores.append(frequency_score)
        details["frequency_score"] = frequency_score

        # Combined score (weighted average)
        weights = [0.3, 0.25, 0.25, 0.2]
        overall_score = sum(s * w for s, w in zip(scores, weights))
        passed = overall_score >= self.threshold

        return CheckResult(
            check_type=LivenessCheck.REPLAY,
            score=overall_score,
            passed=passed,
            threshold=self.threshold,
            details=details,
            message="Replay attack likely" if not passed else "No replay detected",
        )

    def _check_compression_artifacts(self, audio: np.ndarray, sr: int) -> float:
        """Check for lossy compression artifacts."""
        # Compute spectrogram
        n_fft = 2048
        hop = 512
        n_frames = max(1, (len(audio) - n_fft) // hop + 1)

        if n_frames < 2:
            return 0.7  # Not enough data, assume OK

        window = signal.windows.hann(n_fft)
        specs = []

        for i in range(n_frames):
            start = i * hop
            frame = audio[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            frame = frame * window
            spec = np.abs(rfft(frame))
            specs.append(spec)

        specs = np.array(specs)
        avg_spec = np.mean(specs, axis=0)

        # MP3/AAC typically cuts off at ~16kHz and has spectral holes
        freqs = np.linspace(0, sr / 2, len(avg_spec))

        # Check for sudden high-frequency cutoff (compression artifact)
        if sr >= 44100:
            cutoff_region = avg_spec[freqs > 16000]
            if len(cutoff_region) > 0:
                high_freq_energy = np.mean(cutoff_region)
                low_freq_energy = np.mean(avg_spec[freqs < 16000])
                ratio = high_freq_energy / (low_freq_energy + 1e-10)
                if ratio < 0.01:  # Sudden cutoff suggests compression
                    return 0.3

        # Check for spectral banding (compression artifact)
        spec_diff = np.abs(np.diff(avg_spec))
        banding_metric = np.std(spec_diff) / (np.mean(spec_diff) + 1e-10)
        if banding_metric > 5.0:  # Too regular suggests artifacts
            return 0.4

        return 0.8  # Likely natural

    def _check_channel_patterns(self, audio: np.ndarray, sr: int) -> float:
        """Check for channel/environment patterns suggesting replay."""
        # Analyze noise floor
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sqrt(np.mean(frame ** 2)))

        energy = np.array(energy)
        if len(energy) < 10:
            return 0.7

        # Find silence regions
        threshold = np.percentile(energy, 10)
        silence_mask = energy < threshold

        if np.sum(silence_mask) < 5:
            return 0.7  # Not enough silence to analyze

        # Analyze noise in silence regions
        noise_levels = energy[silence_mask]
        noise_std = np.std(noise_levels)
        noise_mean = np.mean(noise_levels)

        # Replayed audio often has very consistent noise floor
        coefficient_of_variation = noise_std / (noise_mean + 1e-10)
        if coefficient_of_variation < 0.1:  # Too consistent
            return 0.4

        return 0.8

    def _check_signal_naturalness(self, audio: np.ndarray, sr: int) -> float:
        """Check if signal has natural characteristics."""
        # Natural speech has certain statistical properties
        if len(audio) < 1000:
            return 0.7

        # Check kurtosis (natural speech has specific range)
        kurt = kurtosis(audio)
        if kurt < 0 or kurt > 20:
            return 0.5  # Unusual distribution

        # Check for micro-variations that exist in live speech
        # but may be smoothed in replays
        diff = np.diff(audio)
        micro_var = np.std(diff)
        if micro_var < 0.001:  # Too smooth
            return 0.4

        # Check for natural zero-crossings
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        if zcr < 0.01 or zcr > 0.5:  # Outside natural range
            return 0.5

        return 0.85

    def _check_frequency_response(self, audio: np.ndarray, sr: int) -> float:
        """Check frequency response for replay artifacts."""
        # Replayed audio often has speaker/mic coloration
        n_fft = min(4096, len(audio))
        if n_fft < 256:
            return 0.7

        window = signal.windows.hann(n_fft)
        spec = np.abs(rfft(audio[:n_fft] * window))
        freqs = np.linspace(0, sr / 2, len(spec))

        # Check for unnatural resonances (speaker artifacts)
        spec_db = 20 * np.log10(spec + 1e-10)
        spec_smooth = np.convolve(spec_db, np.ones(20)/20, mode='same')
        deviation = np.abs(spec_db - spec_smooth)

        # Look for sharp peaks (speaker resonances)
        sharp_peaks = np.sum(deviation > 10)
        if sharp_peaks > len(spec) * 0.1:  # Too many resonances
            return 0.4

        return 0.8


class DeepfakeDetector:
    """
    Detect TTS-generated (deepfake) speech.

    Techniques:
        - Prosody analysis (unnatural timing patterns)
        - Breath detection (TTS often lacks natural breaths)
        - Formant transitions (TTS has smoother transitions)
        - Micro-prosody (subtle pitch/intensity variations)
    """

    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> CheckResult:
        """
        Detect if audio is TTS-generated.

        Args:
            audio: Audio samples
            sample_rate: Sample rate

        Returns:
            CheckResult with deepfake detection score
        """
        if not SCIPY_AVAILABLE:
            return CheckResult(
                check_type=LivenessCheck.DEEPFAKE,
                score=0.5,
                passed=True,
                threshold=self.threshold,
                message="scipy not available, skipping deepfake detection",
            )

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        scores = []
        details = {}

        # 1. Check for natural breathing patterns
        breath_score = self._check_breath_patterns(audio, sample_rate)
        scores.append(breath_score)
        details["breath_score"] = breath_score

        # 2. Check micro-prosody (subtle variations)
        micro_score = self._check_micro_prosody(audio, sample_rate)
        scores.append(micro_score)
        details["micro_prosody_score"] = micro_score

        # 3. Check formant transitions
        formant_score = self._check_formant_transitions(audio, sample_rate)
        scores.append(formant_score)
        details["formant_score"] = formant_score

        # 4. Check for TTS artifacts
        artifact_score = self._check_tts_artifacts(audio, sample_rate)
        scores.append(artifact_score)
        details["artifact_score"] = artifact_score

        # Combined score
        weights = [0.3, 0.25, 0.25, 0.2]
        overall_score = sum(s * w for s, w in zip(scores, weights))
        passed = overall_score >= self.threshold

        return CheckResult(
            check_type=LivenessCheck.DEEPFAKE,
            score=overall_score,
            passed=passed,
            threshold=self.threshold,
            details=details,
            message="Deepfake likely" if not passed else "No deepfake detected",
        )

    def _check_breath_patterns(self, audio: np.ndarray, sr: int) -> float:
        """Check for natural breathing patterns."""
        # Natural speech has breaths every ~5-8 seconds
        duration = len(audio) / sr

        if duration < 3:
            return 0.6  # Too short to analyze

        # Detect potential breath locations (low energy, specific frequency)
        frame_length = int(0.050 * sr)
        hop_length = int(0.010 * sr)

        breath_candidates = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))

            # Breath characteristics: low energy, specific spectral shape
            if energy < 0.05:
                spec = np.abs(rfft(frame))
                freqs = np.linspace(0, sr / 2, len(spec))

                # Breath has energy in 100-1000 Hz range
                breath_band = spec[(freqs > 100) & (freqs < 1000)]
                high_band = spec[freqs > 2000]

                if len(breath_band) > 0 and len(high_band) > 0:
                    if np.mean(breath_band) > np.mean(high_band) * 2:
                        breath_candidates.append(i / sr)

        # Natural speech has ~1 breath per 5-8 seconds
        expected_breaths = duration / 6.0
        actual_breaths = len(breath_candidates)

        if actual_breaths == 0 and duration > 5:
            return 0.3  # No breaths in long audio suggests TTS

        ratio = actual_breaths / (expected_breaths + 0.1)
        if 0.3 < ratio < 3.0:
            return 0.85
        elif ratio < 0.1:
            return 0.4  # Too few breaths
        else:
            return 0.6

    def _check_micro_prosody(self, audio: np.ndarray, sr: int) -> float:
        """Check for natural micro-prosody variations."""
        # Natural speech has subtle jitter and shimmer
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        pitches = []
        intensities = []

        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))
            intensities.append(energy)

            # Simple pitch estimation
            if energy > 0.01:
                corr = np.correlate(frame, frame, mode='full')
                corr = corr[len(corr)//2:]
                min_lag = int(sr / 400)
                max_lag = min(int(sr / 80), len(corr) - 1)

                if max_lag > min_lag:
                    peak_idx = np.argmax(corr[min_lag:max_lag]) + min_lag
                    pitches.append(sr / peak_idx)

        if len(pitches) < 10:
            return 0.6  # Not enough data

        # Check jitter (pitch variation)
        pitch_diff = np.abs(np.diff(pitches))
        jitter = np.mean(pitch_diff) / (np.mean(pitches) + 1e-10)

        # Natural jitter is 0.5-2%
        if 0.005 < jitter < 0.05:
            pitch_score = 0.9
        elif jitter < 0.001:  # Too smooth (TTS-like)
            pitch_score = 0.3
        else:
            pitch_score = 0.6

        # Check shimmer (intensity variation)
        intensity_diff = np.abs(np.diff(intensities))
        voiced_mask = np.array(intensities[:-1]) > 0.01
        if np.sum(voiced_mask) > 0:
            shimmer = np.mean(intensity_diff[voiced_mask]) / (np.mean(np.array(intensities)[:-1][voiced_mask]) + 1e-10)
        else:
            shimmer = 0

        if 0.02 < shimmer < 0.15:
            shimmer_score = 0.9
        elif shimmer < 0.005:  # Too smooth
            shimmer_score = 0.3
        else:
            shimmer_score = 0.6

        return (pitch_score + shimmer_score) / 2

    def _check_formant_transitions(self, audio: np.ndarray, sr: int) -> float:
        """Check formant transitions for naturalness."""
        # TTS often has smoother formant transitions than natural speech
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        formant_tracks = []

        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            if np.sqrt(np.mean(frame ** 2)) < 0.01:
                continue

            # LPC-based formant estimation
            formants = self._estimate_formants_lpc(frame, sr)
            if len(formants) >= 2:
                formant_tracks.append(formants[:2])

        if len(formant_tracks) < 10:
            return 0.6

        formant_tracks = np.array(formant_tracks)

        # Check formant transition speed
        f1_diff = np.abs(np.diff(formant_tracks[:, 0]))
        f2_diff = np.abs(np.diff(formant_tracks[:, 1]))

        # Natural transitions have certain variability
        f1_var = np.std(f1_diff) / (np.mean(f1_diff) + 1e-10)
        f2_var = np.std(f2_diff) / (np.mean(f2_diff) + 1e-10)

        if f1_var < 0.3 or f2_var < 0.3:  # Too regular
            return 0.4

        if f1_var > 3.0 or f2_var > 3.0:  # Too erratic
            return 0.5

        return 0.85

    def _estimate_formants_lpc(self, frame: np.ndarray, sr: int, order: int = 10) -> List[float]:
        """Estimate formants using LPC."""
        # Pre-emphasis
        frame = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])

        # Autocorrelation
        n = len(frame)
        r = np.zeros(order + 1)
        for i in range(order + 1):
            r[i] = np.sum(frame[i:] * frame[:n-i])

        # Levinson-Durbin
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]

        for i in range(1, order + 1):
            sum_val = sum(a[j] * r[i - j] for j in range(i))
            k = -(r[i] + sum_val) / (e + 1e-10)
            a_new = a.copy()
            for j in range(1, i):
                a_new[j] = a[j] + k * a[i - j]
            a_new[i] = k
            a = a_new
            e = e * (1 - k * k)

        # Find roots and extract formants
        roots = np.roots(a)
        formants = []

        for root in roots:
            if np.imag(root) > 0:
                freq = np.abs(np.arctan2(np.imag(root), np.real(root))) * sr / (2 * np.pi)
                if 200 < freq < sr / 2 - 100:
                    formants.append(freq)

        formants.sort()
        return formants

    def _check_tts_artifacts(self, audio: np.ndarray, sr: int) -> float:
        """Check for common TTS artifacts."""
        # TTS can have repetitive patterns, unnatural silences, etc.

        # Check for unnaturally regular pauses
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sqrt(np.mean(frame ** 2)))

        energy = np.array(energy)
        threshold = np.percentile(energy, 20)
        silence = energy < threshold

        # Find silence durations
        silence_durations = []
        current_silence = 0
        for s in silence:
            if s:
                current_silence += 1
            elif current_silence > 0:
                silence_durations.append(current_silence)
                current_silence = 0

        if len(silence_durations) < 2:
            return 0.7

        # TTS often has very regular pause durations
        silence_std = np.std(silence_durations)
        silence_mean = np.mean(silence_durations)
        cv = silence_std / (silence_mean + 1e-10)

        if cv < 0.2:  # Too regular
            return 0.4

        return 0.8


class BreathDetector:
    """Detect natural breathing patterns."""

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> CheckResult:
        """Detect natural breaths and pops."""
        if not SCIPY_AVAILABLE:
            return CheckResult(
                check_type=LivenessCheck.BREATH,
                score=0.5,
                passed=True,
                threshold=self.threshold,
                message="scipy not available",
            )

        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / sample_rate

        if duration < 2:
            return CheckResult(
                check_type=LivenessCheck.BREATH,
                score=0.5,
                passed=True,
                threshold=self.threshold,
                message="Audio too short for breath analysis",
            )

        # Detect breath-like segments
        breath_count = self._count_breaths(audio, sample_rate)
        expected_breaths = max(1, duration / 7.0)

        score = min(1.0, breath_count / expected_breaths)
        passed = score >= self.threshold

        return CheckResult(
            check_type=LivenessCheck.BREATH,
            score=score,
            passed=passed,
            threshold=self.threshold,
            details={"breath_count": breath_count, "expected": expected_breaths},
            message=f"Detected {breath_count} breaths",
        )

    def _count_breaths(self, audio: np.ndarray, sr: int) -> int:
        """Count breath-like events in audio."""
        frame_length = int(0.100 * sr)
        hop_length = int(0.025 * sr)

        count = 0
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy = np.sqrt(np.mean(frame ** 2))

            if 0.005 < energy < 0.03:
                spec = np.abs(rfft(frame))
                freqs = np.linspace(0, sr / 2, len(spec))

                low_energy = np.sum(spec[(freqs > 50) & (freqs < 500)])
                high_energy = np.sum(spec[freqs > 2000])
                total = np.sum(spec)

                if total > 0:
                    low_ratio = low_energy / total
                    high_ratio = high_energy / total

                    if low_ratio > 0.6 and high_ratio < 0.2:
                        count += 1

        return count


class ChannelDetector:
    """Detect channel/environment anomalies."""

    def __init__(self, threshold: float = 0.60):
        self.threshold = threshold

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> CheckResult:
        """Analyze recording channel characteristics."""
        audio = np.asarray(audio, dtype=np.float32)

        # Check for natural room characteristics
        details = {}

        # Noise floor analysis
        energy = np.abs(audio)
        noise_floor = np.percentile(energy, 5)
        signal_peak = np.percentile(energy, 95)
        snr_estimate = 20 * np.log10((signal_peak + 1e-10) / (noise_floor + 1e-10))
        details["snr_estimate"] = snr_estimate

        # Natural recordings have SNR typically 20-50 dB
        if 15 < snr_estimate < 60:
            snr_score = 0.9
        elif snr_estimate > 60:  # Too clean
            snr_score = 0.5
        else:
            snr_score = 0.6

        # Check for natural dynamics
        dynamics = signal_peak / (noise_floor + 1e-10)
        if 100 < dynamics < 10000:
            dynamics_score = 0.85
        else:
            dynamics_score = 0.5

        score = (snr_score + dynamics_score) / 2
        passed = score >= self.threshold

        return CheckResult(
            check_type=LivenessCheck.CHANNEL,
            score=score,
            passed=passed,
            threshold=self.threshold,
            details=details,
        )


class ProsodyDetector:
    """Detect natural prosody patterns."""

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> CheckResult:
        """Analyze prosody for naturalness."""
        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / sample_rate

        if duration < 1:
            return CheckResult(
                check_type=LivenessCheck.PROSODY,
                score=0.5,
                passed=True,
                threshold=self.threshold,
                message="Audio too short",
            )

        # Analyze energy contour
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)

        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sqrt(np.mean(frame ** 2)))

        energy = np.array(energy)
        if len(energy) < 10:
            return CheckResult(
                check_type=LivenessCheck.PROSODY,
                score=0.5,
                passed=True,
                threshold=self.threshold,
            )

        # Check for natural variation
        energy_diff = np.diff(energy)
        variation = np.std(energy_diff) / (np.mean(np.abs(energy_diff)) + 1e-10)

        if 0.5 < variation < 3.0:
            score = 0.85
        elif variation < 0.2:  # Too monotone
            score = 0.4
        else:
            score = 0.6

        passed = score >= self.threshold

        return CheckResult(
            check_type=LivenessCheck.PROSODY,
            score=score,
            passed=passed,
            threshold=self.threshold,
            details={"energy_variation": variation},
        )


class LivenessDetector:
    """
    Main liveness detection orchestrator.

    Combines multiple detection methods for robust anti-spoofing.
    """

    DEFAULT_THRESHOLD = 0.80

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        checks: Optional[List[LivenessCheck]] = None,
    ):
        """
        Initialize liveness detector.

        Args:
            threshold: Overall liveness threshold (0-1)
            checks: List of checks to perform (default: all)
        """
        self.threshold = threshold
        self.checks = checks or list(LivenessCheck)

        # Initialize detectors
        self.replay_detector = ReplayDetector()
        self.deepfake_detector = DeepfakeDetector()
        self.breath_detector = BreathDetector()
        self.channel_detector = ChannelDetector()
        self.prosody_detector = ProsodyDetector()

    def check(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> LivenessResult:
        """
        Perform liveness detection.

        Args:
            audio: Audio samples
            sample_rate: Sample rate

        Returns:
            LivenessResult with detection details
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        results = {}
        details = {}

        # Run enabled checks
        if LivenessCheck.REPLAY in self.checks:
            replay_result = self.replay_detector.detect(audio, sample_rate)
            results["replay"] = replay_result
            details["replay"] = replay_result.details

        if LivenessCheck.DEEPFAKE in self.checks:
            deepfake_result = self.deepfake_detector.detect(audio, sample_rate)
            results["deepfake"] = deepfake_result
            details["deepfake"] = deepfake_result.details

        if LivenessCheck.BREATH in self.checks:
            breath_result = self.breath_detector.detect(audio, sample_rate)
            results["breath"] = breath_result
            details["breath"] = breath_result.details

        if LivenessCheck.CHANNEL in self.checks:
            channel_result = self.channel_detector.detect(audio, sample_rate)
            results["channel"] = channel_result
            details["channel"] = channel_result.details

        if LivenessCheck.PROSODY in self.checks:
            prosody_result = self.prosody_detector.detect(audio, sample_rate)
            results["prosody"] = prosody_result
            details["prosody"] = prosody_result.details

        # Calculate overall score (weighted by check importance)
        weights = {
            "replay": 0.30,
            "deepfake": 0.30,
            "breath": 0.15,
            "channel": 0.10,
            "prosody": 0.15,
        }

        total_weight = sum(weights[k] for k in results.keys())
        overall_score = sum(
            results[k].score * weights[k] / total_weight
            for k in results.keys()
        )

        # Determine status
        replay_detected = "replay" in results and not results["replay"].passed
        deepfake_detected = "deepfake" in results and not results["deepfake"].passed

        if replay_detected:
            status = LivenessStatus.REPLAY_DETECTED
            message = "Replay attack detected"
        elif deepfake_detected:
            status = LivenessStatus.DEEPFAKE_DETECTED
            message = "Deepfake detected"
        elif overall_score >= self.threshold:
            status = LivenessStatus.PASSED
            message = "Liveness verified"
        else:
            status = LivenessStatus.FAILED
            message = f"Liveness check failed (score: {overall_score:.2f})"

        return LivenessResult(
            status=status,
            overall_score=overall_score,
            passed=status == LivenessStatus.PASSED,
            replay_score=results.get("replay", CheckResult(LivenessCheck.REPLAY, 0, False, 0)).score,
            deepfake_score=results.get("deepfake", CheckResult(LivenessCheck.DEEPFAKE, 0, False, 0)).score,
            breath_score=results.get("breath", CheckResult(LivenessCheck.BREATH, 0, False, 0)).score,
            channel_score=results.get("channel", CheckResult(LivenessCheck.CHANNEL, 0, False, 0)).score,
            prosody_score=results.get("prosody", CheckResult(LivenessCheck.PROSODY, 0, False, 0)).score,
            replay_detected=replay_detected,
            deepfake_detected=deepfake_detected,
            details=details,
            message=message,
        )


# Convenience function
def check_liveness(
    audio: np.ndarray,
    sample_rate: int = 24000,
    threshold: float = 0.80,
) -> LivenessResult:
    """
    Check liveness of audio sample.

    Args:
        audio: Audio samples
        sample_rate: Sample rate
        threshold: Liveness threshold

    Returns:
        LivenessResult
    """
    detector = LivenessDetector(threshold=threshold)
    return detector.check(audio, sample_rate)
