"""
VØX Audio Processor
-------------------

Preprocesses audio samples for voice cloning fine-tuning.

Pipeline:
1. Load audio files (various formats)
2. Normalize (sample rate, loudness)
3. Extract mel spectrograms
4. Segment into training chunks
5. Apply augmentations (optional)

Usage:
    from axiom_vox.finetuning import AudioProcessor, AudioSample

    processor = AudioProcessor()
    samples = processor.process_files(
        audio_paths=["voice1.wav", "voice2.mp3"],
        voice_id="my_voice",
    )

    for sample in samples:
        print(f"Duration: {sample.duration_seconds:.1f}s")
        print(f"Mel shape: {sample.mel_spectrogram.shape}")
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict
import hashlib

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False
    torchaudio = None
    T = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

logger = logging.getLogger(__name__)


# ============================================================================
# AUDIO SAMPLE
# ============================================================================

@dataclass
class AudioSample:
    """
    Preprocessed audio sample for training.

    Attributes:
        waveform: Audio waveform tensor
        sample_rate: Sample rate in Hz
        duration_seconds: Duration in seconds
        mel_spectrogram: Mel spectrogram tensor
        transcript: Optional text transcript
        voice_id: Voice identifier
        source_path: Original file path
        segment_index: Index if segmented from longer audio
        file_hash: Hash of original file for deduplication
    """

    waveform: Any  # torch.Tensor
    sample_rate: int
    duration_seconds: float
    mel_spectrogram: Any  # torch.Tensor
    transcript: Optional[str] = None
    voice_id: str = ""
    source_path: str = ""
    segment_index: int = 0
    file_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without tensor data)."""
        return {
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "transcript": self.transcript,
            "voice_id": self.voice_id,
            "source_path": self.source_path,
            "segment_index": self.segment_index,
            "file_hash": self.file_hash,
            "waveform_shape": list(self.waveform.shape) if self.waveform is not None else None,
            "mel_shape": list(self.mel_spectrogram.shape) if self.mel_spectrogram is not None else None,
        }


# ============================================================================
# AUDIO PROCESSOR
# ============================================================================

class AudioProcessor:
    """
    Preprocesses audio samples for voice cloning.

    Pipeline:
    1. Load audio (WAV, MP3, FLAC, OGG, etc.)
    2. Resample to target sample rate
    3. Convert to mono if stereo
    4. Normalize loudness
    5. Extract mel spectrograms
    6. Segment into training-sized chunks
    7. Apply augmentations (optional)
    """

    # Qwen3-TTS native sample rate
    TARGET_SAMPLE_RATE = 24000

    # Segment constraints
    MIN_DURATION = 1.0       # Minimum clip length (seconds)
    MAX_DURATION = 30.0      # Maximum clip length (seconds)
    IDEAL_DURATION = 10.0    # Ideal clip length for training

    # Mel spectrogram settings
    N_MELS = 80
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024

    def __init__(
        self,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        min_duration: float = MIN_DURATION,
        max_duration: float = MAX_DURATION,
        n_mels: int = N_MELS,
        enable_augmentation: bool = True,
    ):
        """
        Initialize audio processor.

        Args:
            target_sample_rate: Target sample rate for all audio
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            n_mels: Number of mel filterbank channels
            enable_augmentation: Whether to apply data augmentation
        """
        if not HAS_TORCHAUDIO:
            logger.warning(
                "torchaudio not available. Audio processing will be limited. "
                "Install with: pip install torchaudio"
            )

        self.target_sample_rate = target_sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.n_mels = n_mels
        self.enable_augmentation = enable_augmentation

        # Initialize mel spectrogram transform
        self._mel_transform = None
        if HAS_TORCHAUDIO:
            self._mel_transform = T.MelSpectrogram(
                sample_rate=target_sample_rate,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                win_length=self.WIN_LENGTH,
                n_mels=n_mels,
                normalized=True,
            )

    def process_files(
        self,
        audio_paths: List[str],
        voice_id: str,
        transcripts: Optional[Dict[str, str]] = None,
    ) -> List[AudioSample]:
        """
        Process multiple audio files into training samples.

        Args:
            audio_paths: List of paths to audio files
            voice_id: Voice identifier
            transcripts: Optional dict mapping file path to transcript

        Returns:
            List of preprocessed AudioSample objects
        """
        if not HAS_TORCHAUDIO:
            raise ImportError(
                "torchaudio is required for audio processing. "
                "Install with: pip install torchaudio"
            )

        samples = []
        transcripts = transcripts or {}

        for path in audio_paths:
            path = str(path)
            try:
                file_samples = self._process_single_file(
                    path,
                    voice_id,
                    transcripts.get(path),
                )
                samples.extend(file_samples)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                continue

        logger.info(
            f"Processed {len(audio_paths)} files into {len(samples)} samples"
        )

        return samples

    def _process_single_file(
        self,
        path: str,
        voice_id: str,
        transcript: Optional[str] = None,
    ) -> List[AudioSample]:
        """Process a single audio file into one or more samples."""
        # Load audio
        waveform, sample_rate = torchaudio.load(path)

        # Compute file hash for deduplication
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

        # Preprocess
        waveform = self._preprocess(waveform, sample_rate)

        # Get duration
        duration = waveform.shape[-1] / self.target_sample_rate

        # Skip if too short
        if duration < self.min_duration:
            logger.warning(
                f"Skipping {path}: duration {duration:.1f}s < min {self.min_duration}s"
            )
            return []

        # Segment if too long
        if duration > self.max_duration:
            segments = self._segment(waveform)
        else:
            segments = [(waveform, duration)]

        # Create samples
        samples = []
        for i, (wave_seg, seg_duration) in enumerate(segments):
            # Extract mel spectrogram
            mel = self._extract_mel(wave_seg)

            # Apply augmentation (if enabled and not the first segment)
            if self.enable_augmentation and i > 0:
                wave_seg, mel = self._augment(wave_seg, mel)

            sample = AudioSample(
                waveform=wave_seg,
                sample_rate=self.target_sample_rate,
                duration_seconds=seg_duration,
                mel_spectrogram=mel,
                transcript=transcript,
                voice_id=voice_id,
                source_path=path,
                segment_index=i,
                file_hash=file_hash,
            )
            samples.append(sample)

        return samples

    def _preprocess(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Preprocess waveform: resample, mono, normalize.

        Args:
            waveform: Raw waveform tensor
            sample_rate: Original sample rate

        Returns:
            Preprocessed waveform
        """
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = T.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Normalize to [-1, 1]
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        # Apply loudness normalization (target -23 LUFS)
        waveform = self._normalize_loudness(waveform)

        return waveform

    def _normalize_loudness(
        self,
        waveform: torch.Tensor,
        target_lufs: float = -23.0,
    ) -> torch.Tensor:
        """
        Normalize loudness to target LUFS.

        Simple RMS-based approximation (true LUFS requires ITU-R BS.1770).
        """
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            # Approximate: -23 LUFS ~ -23 dBFS RMS
            target_rms = 10 ** (target_lufs / 20)
            waveform = waveform * (target_rms / rms)

        # Clip to prevent clipping
        waveform = torch.clamp(waveform, -1.0, 1.0)

        return waveform

    def _segment(
        self,
        waveform: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Segment long audio into training-sized chunks.

        Uses silence detection to find natural break points.

        Returns:
            List of (waveform_segment, duration) tuples
        """
        total_samples = waveform.shape[-1]
        ideal_samples = int(self.IDEAL_DURATION * self.target_sample_rate)
        min_samples = int(self.min_duration * self.target_sample_rate)
        max_samples = int(self.max_duration * self.target_sample_rate)

        segments = []
        start = 0

        while start < total_samples:
            # Determine end point
            end = min(start + max_samples, total_samples)

            # Try to find a natural break point (silence) near the ideal length
            if end - start > ideal_samples:
                # Look for silence in the region after ideal length
                search_start = start + ideal_samples
                search_end = min(search_start + ideal_samples // 2, end)

                break_point = self._find_silence(
                    waveform[:, search_start:search_end],
                    search_start,
                )
                if break_point is not None:
                    end = break_point

            # Extract segment
            segment = waveform[:, start:end]
            duration = (end - start) / self.target_sample_rate

            # Only include if long enough
            if end - start >= min_samples:
                segments.append((segment, duration))

            start = end

        return segments

    def _find_silence(
        self,
        waveform: torch.Tensor,
        offset: int,
        threshold: float = 0.02,
        min_silence_ms: int = 100,
    ) -> Optional[int]:
        """
        Find a silence point in waveform.

        Returns:
            Sample index of silence point (with offset), or None
        """
        min_silence_samples = int(min_silence_ms * self.target_sample_rate / 1000)

        # Compute rolling RMS
        window_size = min_silence_samples
        if waveform.shape[-1] < window_size:
            return None

        # Simple energy-based silence detection
        energy = waveform.abs().squeeze()

        for i in range(0, len(energy) - window_size, window_size // 4):
            window_energy = energy[i:i + window_size].mean()
            if window_energy < threshold:
                return offset + i + window_size // 2

        return None

    def _extract_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram from waveform."""
        if self._mel_transform is None:
            raise RuntimeError("Mel transform not initialized (torchaudio required)")

        mel = self._mel_transform(waveform)

        # Convert to log scale
        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel

    def _augment(
        self,
        waveform: torch.Tensor,
        mel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply data augmentation.

        Augmentations:
        - Small pitch shift (±2 semitones)
        - Time stretching (0.9-1.1x)
        - Additive noise
        """
        if not self.enable_augmentation:
            return waveform, mel

        # Random pitch shift
        if torch.rand(1).item() < 0.3:
            pitch_shift = torch.randint(-2, 3, (1,)).item()
            if pitch_shift != 0:
                try:
                    effects = [["pitch", str(pitch_shift * 100)], ["rate", str(self.target_sample_rate)]]
                    waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                        waveform, self.target_sample_rate, effects
                    )
                except Exception:
                    pass  # sox effects may not be available

        # Add small amount of noise
        if torch.rand(1).item() < 0.2:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
            waveform = torch.clamp(waveform, -1.0, 1.0)

        # Recompute mel after augmentation
        mel = self._extract_mel(waveform)

        return waveform, mel

    def get_total_duration(self, samples: List[AudioSample]) -> float:
        """Get total duration of all samples in seconds."""
        return sum(s.duration_seconds for s in samples)

    def validate_samples(
        self,
        samples: List[AudioSample],
        min_total_duration: float = 60.0,
        max_total_duration: float = 600.0,
    ) -> Tuple[bool, List[str]]:
        """
        Validate samples meet requirements.

        Args:
            samples: List of audio samples
            min_total_duration: Minimum total duration in seconds
            max_total_duration: Maximum total duration in seconds

        Returns:
            (is_valid, list of warning messages)
        """
        warnings = []

        if not samples:
            return False, ["No valid samples found"]

        total_duration = self.get_total_duration(samples)

        if total_duration < min_total_duration:
            warnings.append(
                f"Insufficient audio: {total_duration:.1f}s < minimum {min_total_duration}s"
            )
            return False, warnings

        if total_duration > max_total_duration:
            warnings.append(
                f"Too much audio: {total_duration:.1f}s > maximum {max_total_duration}s. "
                f"Consider using fewer samples."
            )

        # Check for variety (at least 2 unique files)
        unique_sources = len(set(s.source_path for s in samples))
        if unique_sources < 2:
            warnings.append(
                "Only 1 source file. Multiple recordings improve voice quality."
            )

        # Check sample rate consistency
        sample_rates = set(s.sample_rate for s in samples)
        if len(sample_rates) > 1:
            warnings.append(f"Mixed sample rates: {sample_rates}")

        return len(warnings) == 0 or total_duration >= min_total_duration, warnings


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  VØX Audio Processor Demo")
    print("=" * 70)

    if not HAS_TORCHAUDIO:
        print("\ntorchaudio not available. Install with: pip install torchaudio")
        exit(1)

    print("\n1. Creating audio processor...")
    processor = AudioProcessor(
        target_sample_rate=24000,
        min_duration=1.0,
        max_duration=30.0,
        enable_augmentation=False,
    )
    print(f"   Target sample rate: {processor.target_sample_rate}")
    print(f"   Min duration: {processor.min_duration}s")
    print(f"   Max duration: {processor.max_duration}s")

    print("\n2. Creating synthetic test audio...")
    # Create a simple sine wave for testing
    duration = 5.0
    sample_rate = 24000
    t = torch.linspace(0, duration, int(duration * sample_rate))
    waveform = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)  # 440 Hz sine wave
    print(f"   Created synthetic waveform")
    print(f"   Duration: {duration}s")
    print(f"   Shape: {waveform.shape}")

    # Try to save to temp file, fall back to direct processing if torchaudio.save fails
    import tempfile
    test_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            torchaudio.save(f.name, waveform, sample_rate)
            test_file = f.name
        print(f"   Saved test file: {test_file}")
    except (ImportError, RuntimeError) as e:
        print(f"   Note: Could not save audio file ({type(e).__name__})")
        print("   Using direct tensor processing instead")

    print("\n3. Processing audio...")
    if test_file:
        # Process from file
        samples = processor.process_files(
            audio_paths=[test_file],
            voice_id="test_voice",
        )
    else:
        # Process directly from tensor (simulating what process_files does internally)
        resampled = waveform
        if sample_rate != processor.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, processor.target_sample_rate)
            resampled = resampler(waveform)

        # Compute mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=processor.target_sample_rate,
            n_fft=AudioProcessor.N_FFT,
            hop_length=AudioProcessor.HOP_LENGTH,
            n_mels=AudioProcessor.N_MELS,
        )
        mel_spec = mel_transform(resampled)

        samples = [AudioSample(
            waveform=resampled,
            sample_rate=processor.target_sample_rate,
            duration_seconds=duration,
            mel_spectrogram=mel_spec,
            voice_id="test_voice",
            file_hash="demo_synthetic",
        )]

    print(f"   Created {len(samples)} sample(s)")
    for i, sample in enumerate(samples):
        print(f"   Sample {i}:")
        print(f"     Duration: {sample.duration_seconds:.2f}s")
        print(f"     Waveform shape: {sample.waveform.shape}")
        print(f"     Mel shape: {sample.mel_spectrogram.shape}")
        print(f"     Voice ID: {sample.voice_id}")

    print("\n4. Validating samples...")
    is_valid, warnings = processor.validate_samples(
        samples,
        min_total_duration=3.0,  # Lower threshold for demo
    )
    print(f"   Valid: {is_valid}")
    if warnings:
        for w in warnings:
            print(f"   Warning: {w}")

    print("\n5. Total duration...")
    total = processor.get_total_duration(samples)
    print(f"   Total: {total:.2f}s")

    # Cleanup
    if test_file:
        import os
        os.unlink(test_file)

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
