import os
import io
import time
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("axiom_vox.chatterbox")

# =============================================================================
# PRIME VOICE IDENTITY CONSTANTS
# =============================================================================
PRIME_REF_DIR = os.path.expanduser("~/.cache/axiom_vox/prime_voice")
PRIME_REF_DEFAULT = os.path.join(PRIME_REF_DIR, "prime_reference.wav")
PRIME_VECTOR_DEFAULT = os.path.join(PRIME_REF_DIR, "prime_reference.pt")


class ChatterboxEngine:
    """
    Wrapper for Chatterbox Turbo TTS engine.
    Supports high-fidelity voice cloning with low-latency streaming.
    """

    @staticmethod
    def is_available() -> bool:
        """Check if chatterbox-tts is installed."""
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            return True
        except ImportError:
            return False

    def __init__(
        self,
        device: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_vector: Optional[str] = None,
    ):
        """
        Initialize Chatterbox Turbo engine.

        Args:
            device: "cuda", "mps", or "cpu" (auto-detected if None)
            ref_audio: Path to default reference audio for voice cloning.
            ref_vector: Path to precomputed speaker embedding (.pt).
        """
        self._model = None
        self._loaded = False
        self._sample_rate: int = 24000  # updated on load
        self._cached_vector = None

        # Resolve device
        if device:
            self.device = device
        else:
            self.device = self._detect_device()

        # Reference audio/vector for PRIME's voice
        self.ref_audio = ref_audio or PRIME_REF_DEFAULT
        self.ref_vector = ref_vector or (
            PRIME_VECTOR_DEFAULT if os.path.exists(PRIME_VECTOR_DEFAULT) else None
        )

        logger.info(
            f"ChatterboxEngine initialized (device={self.device}, "
            f"ref_audio={self.ref_audio})"
        )

    def _detect_device(self) -> str:
        """Detect the best available processing unit."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self) -> bool:
        """Lazy loader for the Chatterbox model."""
        if self._loaded:
            return True

        logger.info(f"Loading Chatterbox Turbo on {self.device}...")
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            self._loaded = True
            logger.info(f"Chatterbox Turbo loaded successfully (sr={self._sample_rate})")
            return True
        except ImportError:
            logger.error("chatterbox-tts not installed.")
            return False
        except Exception as e:
            logger.error(f"Failed to load Chatterbox: {e}")
            return False

    def _normalize_text(self, text: str) -> str:
        """
        Apply phonetic and structural normalization for branding.
        Ensures 'PRIME' and 'AXIØM' sound fluid.
        """
        # 1. Handle paralinguistics: map [laugh] to better triggers
        text = text.replace("[laugh]", "[laughter]!")
        text = text.replace("[chuckle]", "[laughter]")

        # 2. Phonetic Branding
        # 'AXIØM' -> 'Axyum' (clean flow)
        text = text.replace("AXIØM", "Axyum")
        text = text.replace("AXIOM", "Axyum")

        # 'PRIME' -> 'Pryme' (long I, reduced M emphasis)
        # We only replace uppercase PRIME to protect normal words
        text = text.replace("PRIME", "Pryme")

        # 3. Non-ASCII cleaning
        # Convert Ø to o if missed
        text = text.replace("Ø", "o")
        text = text.replace("ø", "o")

        return text

    def generate(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_vector: Optional[Any] = None,
        out: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> Optional[bytes]:
        """
        Generate speech from text, optionally cloning a reference voice.
        """
        if not self._loaded and not self.load():
            return None

        # Apply branding/phonetic fixes
        text = self._normalize_text(text)

        # Resolve speaker embedding (vectorization)
        import torch
        has_vector = False

        try:
            # 1. Use provided vector (tensor or path)
            if ref_vector is not None:
                if isinstance(ref_vector, (str, Path)):
                    self._model.conds = torch.load(ref_vector, map_location=self.device, weights_only=False)
                else:
                    self._model.conds = ref_vector
                has_vector = True
            # 2. Use cached/default vector
            elif self.ref_vector and not ref_audio:
                if self._cached_vector is None:
                    self._cached_vector = torch.load(self.ref_vector, map_location=self.device, weights_only=False)
                self._model.conds = self._cached_vector
                has_vector = True

            # 3. Fallback to audio reference if no vector resolved
            ref = ref_audio or (self.ref_audio if not has_vector else None)

            if ref and not os.path.exists(ref):
                logger.warning(f"Reference audio not found: {ref}")
                ref = None

            import torchaudio as ta
            t0 = time.time()
            
            # If we have a vector, audio_prompt_path MUST be None
            wav = self._model.generate(
                text,
                audio_prompt_path=ref if not has_vector else None,
                exaggeration=exaggeration,
            )
            gen_time = time.time() - t0

            # wav is a torch tensor [1, samples]
            duration = wav.shape[-1] / self._sample_rate
            logger.info(
                f"Generated {duration:.1f}s audio in {gen_time:.1f}s "
                f"(RTF: {gen_time/duration:.2f}x)"
            )

            # Save to file if requested
            if out:
                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                ta.save(out, wav, self._sample_rate)
                logger.info(f"Saved to {out}")

            # Convert to WAV bytes
            buffer = io.BytesIO()
            ta.save(buffer, wav, self._sample_rate, format="wav")
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Chatterbox generation failed: {e}")
            return None

    def generate_to_file(self, text: str, output_path: str, ref_audio: Optional[str] = None, **kwargs) -> bool:
        """Utility to generate and save to a file."""
        result = self.generate(text, ref_audio=ref_audio, out=output_path, **kwargs)
        return result is not None

    def vectorize(self, audio_path: str, output_path: Optional[str] = None) -> Any:
        """
        Extract and optionally save the speaker embedding from a WAV file.
        
        This 'vectorizes' the voice for sub-200ms latency synthesis.
        """
        if not self._loaded and not self.load():
            return None
            
        try:
            import torch
            # prepare_conditionals populates self._model.conds
            self._model.prepare_conditionals(audio_path)
            vector = self._model.conds
            
            if output_path:
                torch.save(vector, output_path)
                logger.info(f"Voice vectorized and saved to: {output_path}")
                
            return vector
        except Exception as e:
            logger.error(f"Vectorization failed: {e}")
            return None

    @staticmethod
    def setup_prime_reference(audio_path: str, vectorize: bool = True) -> str:
        """
        Copy/link a reference audio file as PRIME's default voice.

        Args:
            audio_path: Path to the reference WAV (5-15 seconds).
            vectorize: If True, also precomputes the .pt vector.

        Returns:
            Path where reference was stored.
        """
        import shutil

        os.makedirs(PRIME_REF_DIR, exist_ok=True)
        dest = PRIME_REF_DEFAULT
        shutil.copy2(audio_path, dest)
        logger.info(f"PRIME reference voice set: {dest}")
        
        if vectorize:
            engine = ChatterboxEngine()
            engine.vectorize(dest, PRIME_VECTOR_DEFAULT)
            
        return dest
