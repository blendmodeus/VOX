"""
AXIOM VOX Transition Processor
------------------------------

Audio transition utilities for multi-voice synthesis.

Provides smooth transitions between voice segments:
- Breath pause: Natural pause with subtle ambient noise
- Crossfade: Overlap fade between segments
- Silence: Clean gap
- Immediate: Direct cut

v0.9.0: Multi-voice Synthesis

Usage:
    from axiom_vox.transition_processor import TransitionProcessor, TransitionStyle

    processor = TransitionProcessor(sample_rate=24000)

    # Generate breath pause
    pause_audio = processor.generate_breath_pause(duration_ms=200)

    # Apply crossfade between segments
    blended = processor.apply_crossfade(audio_a, audio_b, crossfade_ms=150)

    # Create transition based on style
    transition = processor.create_transition(
        TransitionStyle.BREATH_PAUSE,
        previous_audio=audio_a,
        next_audio=audio_b,
    )
"""

import math
import struct
import logging
from enum import Enum
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# TRANSITION STYLE ENUM
# ============================================================================

class TransitionStyle(str, Enum):
    """Voice transition styles."""
    CROSSFADE = "crossfade"      # 150ms overlap fade
    BREATH_PAUSE = "breath_pause"  # 200ms natural pause (default)
    SILENCE = "silence"          # Clean gap
    IMMEDIATE = "immediate"      # Direct cut


# ============================================================================
# TRANSITION CONFIG
# ============================================================================

@dataclass
class TransitionConfig:
    """Configuration for voice transitions."""
    style: TransitionStyle = TransitionStyle.BREATH_PAUSE
    duration_ms: int = 200  # Transition duration
    crossfade_ms: int = 150  # Crossfade overlap
    breath_noise_level: float = 0.02  # Ambient noise level (0-1)
    fade_curve: str = "cosine"  # "linear", "cosine", or "exponential"

    @classmethod
    def for_style(cls, style: TransitionStyle) -> "TransitionConfig":
        """Get default config for a transition style."""
        configs = {
            TransitionStyle.CROSSFADE: cls(
                style=TransitionStyle.CROSSFADE,
                duration_ms=150,
                crossfade_ms=150,
            ),
            TransitionStyle.BREATH_PAUSE: cls(
                style=TransitionStyle.BREATH_PAUSE,
                duration_ms=200,
                breath_noise_level=0.02,
            ),
            TransitionStyle.SILENCE: cls(
                style=TransitionStyle.SILENCE,
                duration_ms=100,
            ),
            TransitionStyle.IMMEDIATE: cls(
                style=TransitionStyle.IMMEDIATE,
                duration_ms=0,
            ),
        }
        return configs.get(style, cls())


# ============================================================================
# TRANSITION RESULT
# ============================================================================

@dataclass
class TransitionResult:
    """Result of a transition operation."""
    audio_bytes: bytes
    duration_ms: int
    style: TransitionStyle
    samples: int

    @property
    def is_empty(self) -> bool:
        """Check if transition produced no audio."""
        return len(self.audio_bytes) == 0 or self.samples == 0


# ============================================================================
# TRANSITION PROCESSOR
# ============================================================================

class TransitionProcessor:
    """
    Processes audio transitions between voice segments.

    Supports various transition styles for natural-sounding
    multi-voice synthesis.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        sample_width: int = 2,  # 16-bit audio
        channels: int = 1,  # Mono
    ):
        """
        Initialize transition processor.

        Args:
            sample_rate: Audio sample rate (Hz)
            sample_width: Bytes per sample (2 for 16-bit)
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        self.bytes_per_sample = sample_width * channels

    def create_transition(
        self,
        style: TransitionStyle,
        config: Optional[TransitionConfig] = None,
        previous_audio: Optional[bytes] = None,
        next_audio: Optional[bytes] = None,
    ) -> TransitionResult:
        """
        Create a transition between audio segments.

        Args:
            style: Transition style
            config: Optional transition config (uses defaults if None)
            previous_audio: Audio ending the previous segment (for crossfade)
            next_audio: Audio starting the next segment (for crossfade)

        Returns:
            TransitionResult with audio bytes
        """
        config = config or TransitionConfig.for_style(style)

        if style == TransitionStyle.IMMEDIATE:
            return TransitionResult(
                audio_bytes=b"",
                duration_ms=0,
                style=style,
                samples=0,
            )

        elif style == TransitionStyle.SILENCE:
            audio = self.generate_silence(config.duration_ms)
            return TransitionResult(
                audio_bytes=audio,
                duration_ms=config.duration_ms,
                style=style,
                samples=len(audio) // self.bytes_per_sample,
            )

        elif style == TransitionStyle.BREATH_PAUSE:
            audio = self.generate_breath_pause(
                duration_ms=config.duration_ms,
                noise_level=config.breath_noise_level,
            )
            return TransitionResult(
                audio_bytes=audio,
                duration_ms=config.duration_ms,
                style=style,
                samples=len(audio) // self.bytes_per_sample,
            )

        elif style == TransitionStyle.CROSSFADE:
            if previous_audio and next_audio:
                audio = self.apply_crossfade(
                    previous_audio,
                    next_audio,
                    crossfade_ms=config.crossfade_ms,
                    fade_curve=config.fade_curve,
                )
                # Crossfade overlaps existing audio, so duration is the overlap
                return TransitionResult(
                    audio_bytes=audio,
                    duration_ms=config.crossfade_ms,
                    style=style,
                    samples=len(audio) // self.bytes_per_sample,
                )
            else:
                # Fallback to breath pause if no audio for crossfade
                logger.warning("Crossfade requested but missing audio segments, using breath pause")
                return self.create_transition(TransitionStyle.BREATH_PAUSE)

        # Fallback
        return TransitionResult(
            audio_bytes=b"",
            duration_ms=0,
            style=style,
            samples=0,
        )

    def generate_silence(self, duration_ms: int) -> bytes:
        """
        Generate silent audio.

        Args:
            duration_ms: Duration in milliseconds

        Returns:
            Silent audio bytes
        """
        num_samples = int(self.sample_rate * duration_ms / 1000)
        return b"\x00" * (num_samples * self.bytes_per_sample)

    def generate_breath_pause(
        self,
        duration_ms: int = 200,
        noise_level: float = 0.02,
    ) -> bytes:
        """
        Generate a natural breath pause.

        Creates subtle ambient noise with fade in/out envelope
        to simulate a natural pause between speakers.

        Args:
            duration_ms: Duration in milliseconds
            noise_level: Base noise level (0.0 to 1.0)

        Returns:
            Breath pause audio bytes
        """
        num_samples = int(self.sample_rate * duration_ms / 1000)
        if num_samples == 0:
            return b""

        # Use a simple linear congruential generator for deterministic noise
        # This avoids importing numpy for a small utility
        samples = []
        seed = 42
        max_amplitude = int(32767 * noise_level)

        # Envelope parameters
        fade_samples = min(num_samples // 4, int(self.sample_rate * 0.05))  # 50ms or 1/4 of duration

        for i in range(num_samples):
            # Generate pseudo-random noise
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            noise = (seed % (2 * max_amplitude + 1)) - max_amplitude

            # Apply envelope
            envelope = 1.0
            if i < fade_samples:
                # Fade in
                envelope = i / fade_samples
            elif i >= num_samples - fade_samples:
                # Fade out
                envelope = (num_samples - i) / fade_samples

            # Apply cosine smoothing to envelope
            envelope = (1 - math.cos(envelope * math.pi)) / 2

            sample = int(noise * envelope)
            samples.append(sample)

        # Pack samples as 16-bit signed integers (little-endian)
        return struct.pack(f"<{len(samples)}h", *samples)

    def apply_crossfade(
        self,
        segment_a: bytes,
        segment_b: bytes,
        crossfade_ms: int = 150,
        fade_curve: str = "cosine",
    ) -> bytes:
        """
        Apply crossfade between two audio segments.

        Creates an overlap region where segment_a fades out
        while segment_b fades in.

        Args:
            segment_a: First audio segment (ending)
            segment_b: Second audio segment (beginning)
            crossfade_ms: Crossfade duration in milliseconds
            fade_curve: Fade curve type ("linear", "cosine", "exponential")

        Returns:
            Crossfaded audio combining end of A and start of B
        """
        crossfade_samples = int(self.sample_rate * crossfade_ms / 1000)

        # Unpack audio samples
        num_samples_a = len(segment_a) // self.bytes_per_sample
        num_samples_b = len(segment_b) // self.bytes_per_sample

        # Need at least crossfade_samples from each segment
        if num_samples_a < crossfade_samples or num_samples_b < crossfade_samples:
            # Not enough audio for crossfade - just concatenate
            logger.warning(
                f"Insufficient audio for {crossfade_ms}ms crossfade "
                f"(A={num_samples_a}, B={num_samples_b} samples)"
            )
            return segment_a + segment_b

        # Extract crossfade regions
        samples_a = struct.unpack(f"<{num_samples_a}h", segment_a)
        samples_b = struct.unpack(f"<{num_samples_b}h", segment_b)

        # Get last crossfade_samples from A and first crossfade_samples from B
        end_a = samples_a[-crossfade_samples:]
        start_b = samples_b[:crossfade_samples]

        # Build crossfade region
        crossfade_samples_out = []
        for i in range(crossfade_samples):
            progress = i / (crossfade_samples - 1) if crossfade_samples > 1 else 1.0

            # Calculate fade factors based on curve type
            if fade_curve == "linear":
                fade_out = 1.0 - progress
                fade_in = progress
            elif fade_curve == "exponential":
                fade_out = math.exp(-3 * progress)
                fade_in = 1.0 - math.exp(-3 * progress)
            else:  # cosine (default)
                fade_out = (1 + math.cos(progress * math.pi)) / 2
                fade_in = (1 - math.cos(progress * math.pi)) / 2

            # Mix samples
            mixed = int(end_a[i] * fade_out + start_b[i] * fade_in)
            # Clamp to 16-bit range
            mixed = max(-32768, min(32767, mixed))
            crossfade_samples_out.append(mixed)

        # Build result: A (minus crossfade) + crossfade + B (minus crossfade)
        pre_crossfade = samples_a[:-crossfade_samples]
        post_crossfade = samples_b[crossfade_samples:]

        all_samples = list(pre_crossfade) + crossfade_samples_out + list(post_crossfade)
        return struct.pack(f"<{len(all_samples)}h", *all_samples)

    def apply_fade_in(
        self,
        audio: bytes,
        fade_ms: int = 50,
        curve: str = "cosine",
    ) -> bytes:
        """
        Apply fade-in to audio segment.

        Args:
            audio: Audio bytes
            fade_ms: Fade duration in milliseconds
            curve: Fade curve type

        Returns:
            Audio with fade-in applied
        """
        fade_samples = int(self.sample_rate * fade_ms / 1000)
        num_samples = len(audio) // self.bytes_per_sample

        if num_samples <= fade_samples:
            fade_samples = num_samples

        samples = list(struct.unpack(f"<{num_samples}h", audio))

        for i in range(fade_samples):
            progress = i / fade_samples if fade_samples > 0 else 1.0

            if curve == "cosine":
                factor = (1 - math.cos(progress * math.pi)) / 2
            elif curve == "exponential":
                factor = 1.0 - math.exp(-3 * progress)
            else:
                factor = progress

            samples[i] = int(samples[i] * factor)

        return struct.pack(f"<{len(samples)}h", *samples)

    def apply_fade_out(
        self,
        audio: bytes,
        fade_ms: int = 50,
        curve: str = "cosine",
    ) -> bytes:
        """
        Apply fade-out to audio segment.

        Args:
            audio: Audio bytes
            fade_ms: Fade duration in milliseconds
            curve: Fade curve type

        Returns:
            Audio with fade-out applied
        """
        fade_samples = int(self.sample_rate * fade_ms / 1000)
        num_samples = len(audio) // self.bytes_per_sample

        if num_samples <= fade_samples:
            fade_samples = num_samples

        samples = list(struct.unpack(f"<{num_samples}h", audio))

        for i in range(fade_samples):
            idx = num_samples - fade_samples + i
            progress = i / fade_samples if fade_samples > 0 else 1.0

            if curve == "cosine":
                factor = (1 + math.cos(progress * math.pi)) / 2
            elif curve == "exponential":
                factor = math.exp(-3 * progress)
            else:
                factor = 1.0 - progress

            samples[idx] = int(samples[idx] * factor)

        return struct.pack(f"<{len(samples)}h", *samples)

    def normalize_volume(
        self,
        audio: bytes,
        target_db: float = -3.0,
    ) -> bytes:
        """
        Normalize audio volume to target dB level.

        Args:
            audio: Audio bytes
            target_db: Target peak level in dB (relative to full scale)

        Returns:
            Volume-normalized audio
        """
        num_samples = len(audio) // self.bytes_per_sample
        if num_samples == 0:
            return audio

        samples = list(struct.unpack(f"<{num_samples}h", audio))

        # Find peak
        peak = max(abs(s) for s in samples) if samples else 1
        if peak == 0:
            return audio

        # Calculate gain
        target_linear = 10 ** (target_db / 20) * 32767
        gain = target_linear / peak

        # Apply gain with clipping
        normalized = [
            max(-32768, min(32767, int(s * gain)))
            for s in samples
        ]

        return struct.pack(f"<{len(normalized)}h", *normalized)

    def get_duration_ms(self, audio: bytes) -> float:
        """Get duration of audio in milliseconds."""
        num_samples = len(audio) // self.bytes_per_sample
        return (num_samples / self.sample_rate) * 1000


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_default_processor: Optional[TransitionProcessor] = None


def get_transition_processor(sample_rate: int = 24000) -> TransitionProcessor:
    """Get or create default transition processor."""
    global _default_processor
    if _default_processor is None or _default_processor.sample_rate != sample_rate:
        _default_processor = TransitionProcessor(sample_rate=sample_rate)
    return _default_processor


def generate_breath_pause(duration_ms: int = 200, sample_rate: int = 24000) -> bytes:
    """Generate a breath pause with default settings."""
    processor = get_transition_processor(sample_rate)
    return processor.generate_breath_pause(duration_ms)


def apply_crossfade(
    segment_a: bytes,
    segment_b: bytes,
    crossfade_ms: int = 150,
    sample_rate: int = 24000,
) -> bytes:
    """Apply crossfade between segments with default settings."""
    processor = get_transition_processor(sample_rate)
    return processor.apply_crossfade(segment_a, segment_b, crossfade_ms)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  AXIOM VOX Transition Processor Demo")
    print("=" * 70)

    processor = TransitionProcessor(sample_rate=24000)

    # Demo 1: Generate breath pause
    print("\n1. Breath Pause Generation:")
    breath = processor.generate_breath_pause(duration_ms=200)
    print(f"   Duration: 200ms")
    print(f"   Bytes: {len(breath)}")
    print(f"   Samples: {len(breath) // 2}")
    print(f"   Actual duration: {processor.get_duration_ms(breath):.1f}ms")

    # Demo 2: Generate silence
    print("\n2. Silence Generation:")
    silence = processor.generate_silence(duration_ms=100)
    print(f"   Duration: 100ms")
    print(f"   Bytes: {len(silence)}")

    # Demo 3: Create different transition styles
    print("\n3. Transition Styles:")
    for style in TransitionStyle:
        result = processor.create_transition(style)
        print(f"   {style.value}: {result.duration_ms}ms, {result.samples} samples")

    # Demo 4: Generate sample audio for crossfade test
    print("\n4. Crossfade Demo:")
    # Create two simple tone segments
    samples_a = []
    samples_b = []
    for i in range(4800):  # 200ms at 24kHz
        # Segment A: 440Hz tone
        samples_a.append(int(16000 * math.sin(2 * math.pi * 440 * i / 24000)))
        # Segment B: 880Hz tone
        samples_b.append(int(16000 * math.sin(2 * math.pi * 880 * i / 24000)))

    audio_a = struct.pack(f"<{len(samples_a)}h", *samples_a)
    audio_b = struct.pack(f"<{len(samples_b)}h", *samples_b)

    print(f"   Segment A: {len(audio_a)} bytes (440Hz tone)")
    print(f"   Segment B: {len(audio_b)} bytes (880Hz tone)")

    crossfaded = processor.apply_crossfade(audio_a, audio_b, crossfade_ms=50)
    print(f"   Crossfaded: {len(crossfaded)} bytes")
    print(f"   Expected: {len(audio_a) + len(audio_b) - (50 * 24 * 2)} bytes")

    # Demo 5: Fade in/out
    print("\n5. Fade In/Out:")
    faded_in = processor.apply_fade_in(audio_a, fade_ms=50)
    faded_out = processor.apply_fade_out(audio_a, fade_ms=50)
    print(f"   Original: {len(audio_a)} bytes")
    print(f"   Fade in applied: {len(faded_in)} bytes")
    print(f"   Fade out applied: {len(faded_out)} bytes")

    # Demo 6: Volume normalization
    print("\n6. Volume Normalization:")
    quiet_samples = [s // 4 for s in samples_a]  # Reduce volume
    quiet_audio = struct.pack(f"<{len(quiet_samples)}h", *quiet_samples)
    normalized = processor.normalize_volume(quiet_audio, target_db=-3.0)
    print(f"   Quiet audio peak: ~{max(abs(s) for s in quiet_samples)}")
    norm_samples = struct.unpack(f"<{len(normalized)//2}h", normalized)
    print(f"   Normalized peak: ~{max(abs(s) for s in norm_samples)}")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
