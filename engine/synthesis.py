"""
VØX Synthesis Engine
--------------------

Speech synthesis with Kokoro TTS (default), Chatterbox Turbo (voice cloning),
and Qwen3-TTS (legacy).

Kokoro: 82M param non-autoregressive TTS, ~0.4x RTF on CPU, 54 voices.
Chatterbox Turbo: 350M param, zero-shot voice cloning, sub-200ms latency.
PRIME voice identity: am_adam:0.65 + bm_daniel:0.35 @ 0.93x speed, +2% pitch.

This module provides the bridge between governed text and audio output.
"""

from __future__ import annotations

import os
import io
import base64
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import Kokoro TTS (default engine)
try:
    from kokoro_onnx import Kokoro as KokoroEngine
    HAS_KOKORO = True
except ImportError:
    HAS_KOKORO = False
    logger.info("kokoro-onnx not installed - install with: pip install kokoro-onnx")

# Try to import Qwen TTS (legacy engine)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from qwen_tts import Qwen3TTSModel
    HAS_QWEN_TTS = True
except ImportError:
    HAS_QWEN_TTS = False

# Try to import Chatterbox Turbo (voice cloning engine)
try:
    from axiom_vox.chatterbox_engine import ChatterboxEngine
    HAS_CHATTERBOX = ChatterboxEngine.is_available()
except ImportError:
    HAS_CHATTERBOX = False

# Try to import audio processing
try:
    import soundfile as sf
    import numpy as np
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    logger.warning("soundfile/numpy not available - audio processing limited")

# =============================================================================
# PRIME Voice Identity (locked)
# =============================================================================
PRIME_VOICE_BLEND = [
    ("am_adam", 0.65),      # Warm, clean American base
    ("bm_daniel", 0.35),    # British crispness, precision
]
PRIME_SPEED = 0.93           # Slightly measured for gravitas
PRIME_PITCH_SHIFT = 1.02     # 2% pitch up (resample ratio)

KOKORO_MODEL_PATH = os.path.expanduser("~/.cache/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.path.expanduser("~/.cache/kokoro/voices-v1.0.bin")


class AudioFormat(str, Enum):
    """Supported audio output formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"


@dataclass
class SynthesisResult:
    """Result from speech synthesis."""
    success: bool
    audio_data: Optional[bytes] = None
    audio_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: int = 24000
    format: AudioFormat = AudioFormat.WAV
    error: Optional[str] = None

    def to_base64(self) -> Optional[str]:
        """Convert audio data to base64 string."""
        if self.audio_data:
            return base64.b64encode(self.audio_data).decode("utf-8")
        return None


@dataclass
class VoiceConfig:
    """Configuration for a voice."""
    voice_id: str
    speaking_rate: float = 1.0      # 0.5 to 2.0
    pitch: float = 0.0              # -1.0 to 1.0 (semitones)
    volume: float = 1.0             # 0.0 to 2.0
    emotion: Optional[str] = None   # Target emotion
    reference_audio: Optional[str] = None  # Path to reference for cloning


class VoxSynthesizer:
    """
    Main synthesis engine for VØX.

    Supports three engines:
    - "kokoro" (default): Kokoro TTS - 82M params, ~0.4x RTF, ONNX runtime
    - "chatterbox": Chatterbox Turbo - 350M params, voice cloning, sub-200ms
    - "qwen": Qwen3-TTS - 0.6B/1.7B params, autoregressive

    Features:
    - Governed synthesis (only speaks approved content)
    - PRIME voice identity (locked blend)
    - Zero-shot voice cloning (Chatterbox engine)
    - Emotion injection + paralinguistic tags
    - Voice cloning with ethical checks (Qwen engine)
    - Streaming support
    """

    # Qwen3-TTS model configurations
    QWEN_MODELS = {
        "small": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "large": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }

    # VoiceDesign models (for natural language voice description)
    VOICE_DESIGN_MODELS = {
        "large": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    }

    # Base models (for voice cloning)
    CLONE_MODELS = {
        "small": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "large": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    }

    # Kokoro voice mappings (voice_id -> kokoro voice name or blend)
    KOKORO_VOICES = {
        "prime_sovereign": PRIME_VOICE_BLEND,
        "default": PRIME_VOICE_BLEND,
        "professional": [("am_adam", 1.0)],
        "conversational": [("am_adam", 0.7), ("am_michael", 0.3)],
        "calm": [("bm_daniel", 1.0)],
        "expert": [("am_adam", 0.65), ("bm_daniel", 0.35)],
        "warm": [("am_adam", 0.8), ("am_michael", 0.2)],
        "announcer": [("am_adam", 0.5), ("bm_george", 0.5)],
    }

    def __init__(
        self,
        engine: str = "kokoro",
        model_size: str = "small",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        adapter_cache_size: int = 10,
    ):
        """
        Initialize the synthesizer.

        Args:
            engine: "kokoro" (default, fast), "chatterbox" (voice cloning), or "qwen" (legacy)
            model_size: "small" (0.6B) or "large" (1.7B) - Qwen engine only
            device: "cuda", "mps", or "cpu" (auto-detected if None)
            cache_dir: Directory to cache models
            adapter_cache_size: Max number of LoRA adapters to keep in memory
        """
        self.engine = engine
        self.model_size = model_size
        self.model_id = self.QWEN_MODELS.get(model_size, self.QWEN_MODELS["small"])
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/axiom_vox")

        # Detect device (for Qwen engine)
        if device:
            self.device = device
        elif HAS_TORCH:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"

        # Model state (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._loaded = False

        # Kokoro-specific state
        self._kokoro = None
        self._kokoro_voices: Dict[str, Any] = {}  # cached blended voice vectors

        # Chatterbox-specific state
        self._chatterbox = None

        # LoRA adapter management (Qwen engine)
        self._adapter_cache: Dict[str, Any] = {}  # voice_id -> VoxLoRAAdapter
        self._adapter_cache_size = adapter_cache_size
        self._active_adapter: Optional[str] = None

        # Auto-fallback chain: requested engine → kokoro → chatterbox → qwen
        if self.engine == "chatterbox" and not HAS_CHATTERBOX:
            if HAS_KOKORO:
                logger.info("Chatterbox not available, falling back to Kokoro")
                self.engine = "kokoro"
            elif HAS_QWEN_TTS:
                logger.info("Chatterbox not available, falling back to Qwen")
                self.engine = "qwen"
            else:
                logger.warning("No TTS engines available")
        elif self.engine == "kokoro" and not HAS_KOKORO:
            if HAS_CHATTERBOX:
                logger.info("Kokoro not available, falling back to Chatterbox")
                self.engine = "chatterbox"
            elif HAS_QWEN_TTS:
                logger.info("Kokoro not available, falling back to Qwen")
                self.engine = "qwen"
            else:
                logger.warning("No TTS engines available")

    def _ensure_loaded(self) -> bool:
        """Ensure model is loaded."""
        if self._loaded:
            return True

        if self.engine == "kokoro":
            return self._load_kokoro()
        elif self.engine == "chatterbox":
            return self._load_chatterbox()
        else:
            return self._load_qwen()

    def _load_kokoro(self) -> bool:
        """Load Kokoro TTS engine."""
        if not HAS_KOKORO:
            logger.error("Cannot load Kokoro - kokoro-onnx not installed")
            return False

        try:
            logger.info("Loading Kokoro TTS...")
            self._kokoro = KokoroEngine(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
            self._loaded = True
            logger.info("Kokoro TTS loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load Kokoro: {e}")
            return False

    def _build_kokoro_voice(self, voice_id: str) -> "np.ndarray":
        """Build or retrieve a blended Kokoro voice vector."""
        if voice_id in self._kokoro_voices:
            return self._kokoro_voices[voice_id]

        # Look up blend spec, default to PRIME voice
        blend = self.KOKORO_VOICES.get(voice_id, PRIME_VOICE_BLEND)

        result = None
        for name, weight in blend:
            style = self._kokoro.get_voice_style(name)
            result = style * weight if result is None else result + style * weight

        self._kokoro_voices[voice_id] = result
        return result

    def _load_chatterbox(self) -> bool:
        """Load Chatterbox Turbo engine."""
        if not HAS_CHATTERBOX:
            logger.error("Cannot load Chatterbox — not installed")
            return False

        try:
            self._chatterbox = ChatterboxEngine(device=self.device)
            loaded = self._chatterbox.load()
            if loaded:
                self._loaded = True
                logger.info("Chatterbox Turbo loaded successfully")
            return loaded
        except Exception as e:
            logger.error(f"Failed to load Chatterbox: {e}")
            return False

    def _load_qwen(self) -> bool:
        """Load Qwen3-TTS engine."""
        if not HAS_TORCH:
            logger.error("Cannot load model - PyTorch not installed")
            return False

        if not HAS_QWEN_TTS:
            logger.error("Cannot load model - qwen-tts not installed. Run: pip install qwen-tts")
            return False

        try:
            logger.info(f"Loading {self.model_id} on {self.device}...")

            # Determine dtype based on device
            if self.device == "cpu":
                dtype = torch.float32
            else:
                dtype = torch.bfloat16

            # Determine attention implementation
            attn_impl = "eager"  # Default for MPS/CPU
            if self.device == "cuda":
                try:
                    import flash_attn  # noqa: F401
                    attn_impl = "flash_attention_2"
                except ImportError:
                    attn_impl = "sdpa"

            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device if self.device != "mps" else None,
                dtype=dtype,
                attn_implementation=attn_impl,
            )

            # For MPS, move model manually
            if self.device == "mps":
                self._model = self._model.to("mps")

            self._loaded = True
            logger.info(f"Qwen3-TTS loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def load_adapter(self, voice_id: str) -> bool:
        """
        Load a LoRA adapter for a cloned voice.

        Args:
            voice_id: Voice identifier (should start with "clone_")

        Returns:
            True if adapter loaded successfully
        """
        # Check if already in cache
        if voice_id in self._adapter_cache:
            self._active_adapter = voice_id
            logger.debug(f"Using cached adapter for {voice_id}")
            return True

        # Ensure base model is loaded
        if not self._ensure_loaded():
            logger.error("Cannot load adapter - base model not available")
            return False

        # Look up adapter path from database
        try:
            from axiom_vox.persistence import get_database
            db = get_database()
            adapter_info = db.get_adapter(voice_id)

            if not adapter_info:
                logger.warning(f"No adapter found for voice: {voice_id}")
                return False

            # Load adapter
            from axiom_vox.finetuning.lora_adapter import VoxLoRAAdapter
            adapter = VoxLoRAAdapter.from_file(
                adapter_info["adapter_path"],
                self._model,
            )

            # LRU eviction if cache is full
            if len(self._adapter_cache) >= self._adapter_cache_size:
                oldest = next(iter(self._adapter_cache))
                del self._adapter_cache[oldest]
                logger.debug(f"Evicted adapter from cache: {oldest}")

            self._adapter_cache[voice_id] = adapter
            self._active_adapter = voice_id

            # Update last used timestamp
            db.update_adapter_last_used(voice_id)

            logger.info(f"Loaded adapter for voice: {voice_id}")
            return True

        except ImportError as e:
            logger.error(f"Fine-tuning module not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load adapter for {voice_id}: {e}")
            return False

    def unload_adapter(self, voice_id: Optional[str] = None) -> None:
        """
        Unload an adapter from cache.

        Args:
            voice_id: Specific voice to unload, or None to unload active
        """
        voice_to_unload = voice_id or self._active_adapter

        if voice_to_unload and voice_to_unload in self._adapter_cache:
            del self._adapter_cache[voice_to_unload]
            if self._active_adapter == voice_to_unload:
                self._active_adapter = None
            logger.debug(f"Unloaded adapter: {voice_to_unload}")

    def get_active_adapter(self) -> Optional[str]:
        """Get the currently active adapter voice ID."""
        return self._active_adapter

    def list_cached_adapters(self) -> list:
        """List all adapters currently in cache."""
        return list(self._adapter_cache.keys())

    def synthesize(
        self,
        text: str,
        voice: Optional[VoiceConfig] = None,
        output_format: AudioFormat = AudioFormat.WAV,
        output_path: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice configuration
            output_format: Audio output format
            output_path: Optional path to save audio

        Returns:
            SynthesisResult with audio data or error
        """
        voice = voice or VoiceConfig(voice_id="default")

        # Auto-load adapter for cloned voices
        if voice.voice_id.startswith("clone_"):
            if not self.load_adapter(voice.voice_id):
                return SynthesisResult(
                    success=False,
                    error=f"Failed to load adapter for cloned voice: {voice.voice_id}",
                )

        # Try real synthesis
        if self._ensure_loaded():
            return self._synthesize_real(text, voice, output_format, output_path)

        # Fall back to placeholder
        return self._synthesize_placeholder(text, voice, output_format, output_path)

    def synthesize_ssml(
        self,
        ssml: str,
        voice: Optional[VoiceConfig] = None,
        output_format: AudioFormat = AudioFormat.WAV,
        output_path: Optional[str] = None,
    ) -> SynthesisResult:
        """
        Synthesize speech from SSML markup.

        Parses SSML and applies prosody controls (pauses, emphasis, rate, pitch)
        to the synthesized audio.

        Args:
            ssml: SSML markup string
            voice: Voice configuration (some params may be overridden by SSML)
            output_format: Audio output format
            output_path: Optional path to save audio

        Returns:
            SynthesisResult with audio data or error
        """
        try:
            from axiom_vox.ssml import SSMLParser
        except ImportError:
            logger.warning("SSML module not available, treating as plain text")
            # Strip SSML tags as fallback
            import re
            plain_text = re.sub(r'<[^>]+>', ' ', ssml)
            plain_text = ' '.join(plain_text.split())
            return self.synthesize(plain_text, voice, output_format, output_path)

        parser = SSMLParser()
        doc, warnings = parser.parse(ssml)

        if warnings:
            logger.warning(f"SSML parse warnings: {warnings}")

        voice = voice or VoiceConfig(voice_id="default")

        # Apply SSML prosody to voice config
        voice = self._apply_ssml_to_voice(doc, voice)

        # Synthesize with modified voice config
        return self.synthesize(doc.plain_text, voice, output_format, output_path)

    def _apply_ssml_to_voice(
        self,
        ssml_doc: "SSMLDocument",
        voice: VoiceConfig,
    ) -> VoiceConfig:
        """Apply SSML document settings to voice config."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()

        # Create a copy of voice config with SSML overrides
        new_voice = VoiceConfig(
            voice_id=voice.voice_id,
            speaking_rate=voice.speaking_rate,
            pitch=voice.pitch,
            volume=voice.volume,
            emotion=voice.emotion,
            reference_audio=voice.reference_audio,
        )

        # Apply first prosody span's values if present
        if ssml_doc.prosody_spans:
            first_prosody = ssml_doc.prosody_spans[0]

            if first_prosody.rate:
                new_voice.speaking_rate = parser.parse_rate(first_prosody.rate)

            if first_prosody.pitch:
                new_voice.pitch = parser.parse_pitch(first_prosody.pitch)

            # Note: volume handling would require audio post-processing
            # which is beyond the scope of VoiceConfig

        return new_voice

    def _synthesize_real(
        self,
        text: str,
        voice: VoiceConfig,
        output_format: AudioFormat,
        output_path: Optional[str],
    ) -> SynthesisResult:
        """Route to the active engine for synthesis."""
        if self.engine == "kokoro":
            return self._synthesize_kokoro(text, voice, output_format, output_path)
        elif self.engine == "chatterbox":
            return self._synthesize_chatterbox(text, voice, output_format, output_path)
        else:
            return self._synthesize_qwen(text, voice, output_format, output_path)

    def _synthesize_kokoro(
        self,
        text: str,
        voice: VoiceConfig,
        output_format: AudioFormat,
        output_path: Optional[str],
    ) -> SynthesisResult:
        """Synthesis using Kokoro TTS."""
        try:
            import numpy as np

            # Build blended voice vector
            kokoro_voice = self._build_kokoro_voice(voice.voice_id)

            # Determine speed from voice config
            speed = PRIME_SPEED
            if voice.speaking_rate != 1.0:
                speed = PRIME_SPEED * voice.speaking_rate

            # Generate audio
            audio_array, sr = self._kokoro.create(
                text, voice=kokoro_voice, speed=speed,
            )

            # Apply pitch shift via sample rate adjustment
            effective_sr = int(sr * PRIME_PITCH_SHIFT)

            # Apply voice config post-processing (volume)
            audio_array = self._apply_voice_config(audio_array, voice)

            # Convert to bytes at the pitch-shifted sample rate
            audio_data = self._array_to_bytes(
                audio_array, output_format, sample_rate=effective_sr,
            )

            # Duration accounts for pitch shift
            duration = len(audio_array) / effective_sr

            if output_path:
                with open(output_path, "wb") as f:
                    f.write(audio_data)

            return SynthesisResult(
                success=True,
                audio_data=audio_data,
                audio_path=output_path,
                duration_seconds=duration,
                sample_rate=effective_sr,
                format=output_format,
            )

        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}")
            return SynthesisResult(success=False, error=str(e))

    def _synthesize_chatterbox(
        self,
        text: str,
        voice: VoiceConfig,
        output_format: AudioFormat,
        output_path: Optional[str],
    ) -> SynthesisResult:
        """Synthesis using Chatterbox Turbo (voice cloning)."""
        try:
            import numpy as np

            # Use reference_audio from voice config, or PRIME's default
            ref_audio = voice.reference_audio

            # Generate audio bytes (WAV)
            wav_bytes = self._chatterbox.generate(
                text,
                ref_audio=ref_audio,
                out=output_path,
            )

            if wav_bytes is None:
                return SynthesisResult(
                    success=False,
                    error="Chatterbox generation returned None",
                )

            # Parse duration from WAV bytes
            import soundfile as sf
            audio_array, sr = sf.read(io.BytesIO(wav_bytes))
            duration = len(audio_array) / sr

            # Apply volume from voice config
            if voice.volume != 1.0:
                audio_array = audio_array * voice.volume
                wav_bytes = self._array_to_bytes(
                    audio_array, output_format, sample_rate=sr,
                )

            return SynthesisResult(
                success=True,
                audio_data=wav_bytes,
                audio_path=output_path,
                duration_seconds=duration,
                sample_rate=sr,
                format=output_format,
            )

        except Exception as e:
            logger.error(f"Chatterbox synthesis failed: {e}")
            return SynthesisResult(success=False, error=str(e))

    def _synthesize_qwen(
        self,
        text: str,
        voice: VoiceConfig,
        output_format: AudioFormat,
        output_path: Optional[str],
    ) -> SynthesisResult:
        """Synthesis using Qwen3-TTS (legacy)."""
        try:
            import numpy as np

            # Build instruction from voice config (emotion, rate, etc.)
            instruct = self._build_instruct(voice)

            # Map voice_id to speaker name or use default
            speaker = self._resolve_speaker(voice.voice_id)

            # Generate audio using Qwen3-TTS native API
            wavs, sr = self._model.generate_custom_voice(
                text=text,
                language="English",
                speaker=speaker,
                instruct=instruct,
            )

            audio_array = wavs[0]  # First (only) waveform

            # Apply voice config post-processing (volume, etc.)
            audio_array = self._apply_voice_config(audio_array, voice)

            # Convert to bytes
            audio_data = self._array_to_bytes(audio_array, output_format, sample_rate=sr)

            # Calculate duration
            duration = len(audio_array) / sr

            # Save if path provided
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(audio_data)

            return SynthesisResult(
                success=True,
                audio_data=audio_data,
                audio_path=output_path,
                duration_seconds=duration,
                sample_rate=sr,
                format=output_format,
            )

        except Exception as e:
            logger.error(f"Qwen synthesis failed: {e}")
            return SynthesisResult(success=False, error=str(e))

    def _build_instruct(self, voice: VoiceConfig) -> str:
        """Build instruction string from voice config for Qwen3-TTS."""
        parts = []

        if voice.emotion:
            parts.append(f"Speak with a {voice.emotion} tone.")

        if voice.speaking_rate < 0.9:
            parts.append("Speak slowly and deliberately.")
        elif voice.speaking_rate > 1.1:
            parts.append("Speak at a brisk, energetic pace.")

        if voice.pitch < -0.5:
            parts.append("Use a deep, low-pitched voice.")
        elif voice.pitch > 0.5:
            parts.append("Use a slightly higher pitch.")

        return " ".join(parts) if parts else ""

    # Available Qwen3-TTS speakers (legacy engine):
    # Male:   aiden, dylan, eric, ryan, uncle_fu
    # Female: ono_anna, serena, sohee, vivian
    QWEN_SPEAKERS = ["aiden", "dylan", "eric", "ono_anna", "ryan",
                     "serena", "sohee", "uncle_fu", "vivian"]

    def _resolve_speaker(self, voice_id: str) -> str:
        """Map VØX voice_id to a Qwen3-TTS speaker name (Qwen engine only)."""
        speaker_map = {
            "prime_sovereign": "eric",
            "default": "eric",
            "professional": "eric",
            "conversational": "aiden",
            "calm": "dylan",
            "expert": "eric",
            "warm": "aiden",
            "announcer": "ryan",
        }
        return speaker_map.get(voice_id, "eric")

    def _synthesize_placeholder(
        self,
        text: str,
        voice: VoiceConfig,
        output_format: AudioFormat,
        output_path: Optional[str],
    ) -> SynthesisResult:
        """
        Placeholder synthesis when model not available.

        Returns a simple tone or silence to indicate synthesis would occur.
        """
        if not HAS_AUDIO:
            return SynthesisResult(
                success=False,
                error="Audio libraries not available. Install: pip install soundfile numpy",
            )

        # Generate a simple placeholder tone
        sample_rate = 24000
        duration = min(len(text) / 15, 10.0)  # Rough estimate, max 10 seconds

        # Generate silence with small beep at start
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Small beep at start to indicate placeholder
        beep_duration = 0.1
        beep_samples = int(sample_rate * beep_duration)
        audio = np.zeros_like(t)
        audio[:beep_samples] = 0.3 * np.sin(2 * np.pi * 440 * t[:beep_samples])

        # Fade out beep
        fade_samples = int(sample_rate * 0.02)
        audio[beep_samples-fade_samples:beep_samples] *= np.linspace(1, 0, fade_samples)

        # Convert to bytes
        audio_data = self._array_to_bytes(audio, output_format)

        if output_path:
            with open(output_path, "wb") as f:
                f.write(audio_data)

        return SynthesisResult(
            success=True,
            audio_data=audio_data,
            audio_path=output_path,
            duration_seconds=duration,
            format=output_format,
            error="PLACEHOLDER: Real model not loaded",
        )

    def _build_prompt(self, text: str, voice: VoiceConfig) -> str:
        """Build prompt string (legacy compatibility)."""
        return text

    def _apply_voice_config(
        self,
        audio: "np.ndarray",
        voice: VoiceConfig,
    ) -> "np.ndarray":
        """Apply voice configuration (pitch, volume, etc.)."""
        if not HAS_AUDIO or audio is None:
            return audio

        # Apply volume
        audio = audio * voice.volume

        # Pitch shifting would require librosa or similar
        # Speed adjustment would require resampling

        return audio

    def _array_to_bytes(
        self,
        audio: "np.ndarray",
        output_format: AudioFormat,
        sample_rate: int = 24000,
    ) -> bytes:
        """Convert numpy array to audio bytes."""
        if not HAS_AUDIO:
            return b""

        buffer = io.BytesIO()

        # Ensure float32 range
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        # Write to buffer
        format_str = output_format.value.upper()
        if format_str == "MP3":
            format_str = "WAV"  # soundfile doesn't support MP3 directly

        sf.write(buffer, audio, sample_rate, format=format_str)

        return buffer.getvalue()

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[VoiceConfig] = None,
        chunk_size: int = 4096,
        use_true_streaming: bool = True,
    ) -> AsyncIterator[bytes]:
        """
        Stream audio chunks as they're generated.

        Args:
            text: Text to synthesize
            voice: Voice configuration
            chunk_size: Size of each audio chunk in bytes
            use_true_streaming: If True, uses sentence-level streaming.
                               If False, generates all then chunks (legacy behavior).

        Yields:
            Audio chunks as bytes for real-time playback.
        """
        import asyncio

        voice = voice or VoiceConfig(voice_id="default")

        # Auto-load adapter for cloned voices
        if voice.voice_id.startswith("clone_"):
            if not self.load_adapter(voice.voice_id):
                logger.error(f"Failed to load adapter for {voice.voice_id}")
                return

        if not use_true_streaming:
            # Legacy behavior: generate all, then chunk
            result = self.synthesize(text, voice)
            if not result.success or not result.audio_data:
                return

            data = result.audio_data
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
            return

        # True streaming: sentence-by-sentence synthesis
        try:
            from axiom_vox.streaming import SentenceSegmenter
        except ImportError:
            # Fallback to legacy if streaming module not available
            logger.warning("Streaming module not available, using legacy chunking")
            result = self.synthesize(text, voice)
            if not result.success or not result.audio_data:
                return
            data = result.audio_data
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
            return

        sentences = SentenceSegmenter.segment(text)
        if not sentences:
            return

        for sentence in sentences:
            # Synthesize this sentence
            result = self.synthesize(sentence, voice)

            if not result.success or not result.audio_data:
                continue

            # Yield chunks from this sentence
            data = result.audio_data
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]
                # Small delay to enable interleaving
                await asyncio.sleep(0.001)

    async def synthesize_stream_with_callbacks(
        self,
        text: str,
        voice: Optional[VoiceConfig] = None,
        chunk_size: int = 4096,
        on_sentence_start: Optional[Callable[[int, str], None]] = None,
        on_sentence_end: Optional[Callable[[int], None]] = None,
        on_chunk: Optional[Callable[[int, bytes], None]] = None,
    ) -> AsyncIterator[bytes]:
        """
        Stream with callbacks for fine-grained control.

        Useful for:
        - UI synchronization (highlight current sentence)
        - Analytics (track chunk delivery)
        - Custom buffering strategies

        Args:
            text: Text to synthesize
            voice: Voice configuration
            chunk_size: Size of each audio chunk in bytes
            on_sentence_start: Called when starting a new sentence (index, text)
            on_sentence_end: Called when finishing a sentence (index)
            on_chunk: Called for each chunk (index, data)

        Yields:
            Audio chunks as bytes for real-time playback.
        """
        import asyncio

        try:
            from axiom_vox.streaming import SentenceSegmenter
        except ImportError:
            logger.warning("Streaming module not available")
            return

        voice = voice or VoiceConfig(voice_id="default")
        sentences = SentenceSegmenter.segment(text)
        chunk_index = 0

        for sentence_idx, sentence in enumerate(sentences):
            if on_sentence_start:
                on_sentence_start(sentence_idx, sentence)

            result = self.synthesize(sentence, voice)

            if result.success and result.audio_data:
                data = result.audio_data
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    if on_chunk:
                        on_chunk(chunk_index, chunk)
                    yield chunk
                    chunk_index += 1
                    await asyncio.sleep(0.001)

            if on_sentence_end:
                on_sentence_end(sentence_idx)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_synthesizer: Optional[VoxSynthesizer] = None


def get_synthesizer(
    engine: str = "kokoro",
    model_size: str = "small",
) -> VoxSynthesizer:
    """Get or create the global synthesizer."""
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = VoxSynthesizer(engine=engine, model_size=model_size)
    return _synthesizer


def synthesize(
    text: str,
    voice_id: str = "default",
    emotion: Optional[str] = None,
    output_path: Optional[str] = None,
) -> SynthesisResult:
    """
    Quick synthesis function.

    Args:
        text: Text to speak
        voice_id: Voice to use
        emotion: Target emotion
        output_path: Optional save path

    Returns:
        SynthesisResult with audio
    """
    synth = get_synthesizer()
    voice = VoiceConfig(voice_id=voice_id, emotion=emotion)
    return synth.synthesize(text, voice, output_path=output_path)


# ============================================================================
# CLI DEMO
# ============================================================================

if __name__ == "__main__":
    import sys as _sys

    print("=" * 70)
    print("  VØX Synthesis Engine Demo")
    print("=" * 70)

    engine = "qwen" if "--qwen" in _sys.argv else "kokoro"
    synth = VoxSynthesizer(engine=engine)

    print(f"\nEngine: {synth.engine}")
    if synth.engine == "qwen":
        print(f"Device: {synth.device}")
        print(f"Model: {synth.model_id}")

    # Test synthesis
    text = "Hello, this is a test of the VØX text to speech system."
    print(f"\nSynthesizing: {text}")

    result = synth.synthesize(text)

    if result.success:
        print(f"Success!")
        print(f"  Duration: {result.duration_seconds:.2f}s")
        print(f"  Sample rate: {result.sample_rate}")
        print(f"  Format: {result.format.value}")
        print(f"  Size: {len(result.audio_data)} bytes")
        if result.error:
            print(f"  Note: {result.error}")
    else:
        print(f"Failed: {result.error}")
