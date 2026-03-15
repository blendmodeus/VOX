"""
VØX Transcriber
---------------

Core speech-to-text engine wrapping faster-whisper.
Replaces the raw WhisperModel usage in vox_api.py with
proper model management, device detection, and structured results.

Usage:
    from axiom_vox.stt import VoxTranscriber, TranscriptionConfig

    transcriber = VoxTranscriber(model_size="base")
    result = transcriber.transcribe("/path/to/audio.wav")
    print(result.text)
    print(result.segments[0].words)

    # Or use convenience function:
    from axiom_vox.stt import transcribe
    result = transcribe("/path/to/audio.wav")
"""

from __future__ import annotations

import logging
import os
import time
import tempfile
from typing import Optional, Tuple, List

from axiom_vox.stt.models import (
    TranscriptionConfig,
    TranscriptionResult,
    TranscriptionSegment,
    WordTimestamp,
    STTModelSize,
    STTModelInfo,
    STTDevice,
    AVAILABLE_MODELS,
    get_model_info,
)

logger = logging.getLogger(__name__)

# Guard for faster-whisper availability
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False
    logger.warning("faster-whisper not installed — STT will use fallback mode")


def detect_device() -> str:
    """Auto-detect best compute device.

    faster-whisper/CTranslate2 doesn't support MPS yet,
    so MPS falls back to CPU with int8 quantization.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            # MPS detected but CTranslate2 doesn't support it
            logger.info("MPS detected but CTranslate2 uses CPU. Using int8 quantization.")
            return "cpu"
    except ImportError:
        pass
    return "cpu"


def _compute_type_for_device(device: str) -> str:
    """Select optimal compute type for device."""
    if device == "cuda":
        return "float16"
    return "int8"


class VoxTranscriber:
    """AXIØM-governed speech-to-text transcriber.

    Wraps faster-whisper with:
    - Dynamic model loading/switching
    - Device auto-detection
    - Structured results with segments + word timestamps
    - Confidence scoring
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: Optional[str] = None,
    ):
        self._model_size = model_size
        self._device = detect_device() if device == "auto" else device
        self._compute_type = compute_type or _compute_type_for_device(self._device)
        self._model: Optional[WhisperModel] = None
        self._loaded = False

    @property
    def model_size(self) -> str:
        return self._model_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _ensure_model(self) -> WhisperModel:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return self._model

        if not HAS_FASTER_WHISPER:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Install with: pip install faster-whisper"
            )

        logger.info(
            f"Loading faster-whisper model: {self._model_size} "
            f"(device={self._device}, compute_type={self._compute_type})"
        )

        # CUDA gets distil-large-v3 for best speed/quality
        actual_model = self._model_size
        if self._device == "cuda" and self._model_size in ("large-v3",):
            actual_model = "distil-large-v3"
            logger.info(f"CUDA detected — using distil-large-v3 for optimal speed")

        self._model = WhisperModel(
            actual_model,
            device=self._device,
            compute_type=self._compute_type,
        )
        self._loaded = True
        logger.info(f"Model loaded: {actual_model}")
        return self._model

    def load_model(self, size: str) -> None:
        """Switch to a different model size (used by frontend MODEL button)."""
        if size == self._model_size and self._loaded:
            return

        self.unload_model()
        self._model_size = size
        self._ensure_model()

    def unload_model(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info("Model unloaded")

            # Attempt GPU memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe(
        self,
        audio_path: str,
        config: Optional[TranscriptionConfig] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file (wav, mp3, ogg, flac, webm, m4a)
            config: Optional transcription config

        Returns:
            TranscriptionResult with text, segments, word timestamps
        """
        config = config or TranscriptionConfig()

        # If config requests a different model, switch
        if config.model_size.value != self._model_size:
            self.load_model(config.model_size.value)

        model = self._ensure_model()
        t0 = time.time()

        segments_iter, info = model.transcribe(
            audio_path,
            language=config.language,
            beam_size=config.beam_size,
            word_timestamps=config.word_timestamps,
            vad_filter=config.vad_filter,
            vad_parameters=dict(
                threshold=config.vad_threshold,
                min_silence_duration_ms=config.min_silence_duration_ms,
            ),
            condition_on_previous_text=config.condition_on_previous_text,
            temperature=config.temperature,
        )

        # Build structured segments
        result_segments: List[TranscriptionSegment] = []
        full_text_parts: List[str] = []

        for seg in segments_iter:
            words = []
            if config.word_timestamps and seg.words:
                words = [
                    WordTimestamp(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        probability=w.probability,
                    )
                    for w in seg.words
                ]

            # Map avg_logprob to a 0-1 confidence score
            # avg_logprob is typically -0.0 (perfect) to -1.0+ (poor)
            confidence = max(0.0, min(1.0, 1.0 + seg.avg_logprob))

            result_segments.append(TranscriptionSegment(
                id=seg.id,
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                confidence=confidence,
                words=words,
                no_speech_prob=seg.no_speech_prob,
            ))
            full_text_parts.append(seg.text)

        elapsed = time.time() - t0
        full_text = "".join(full_text_parts).strip()

        return TranscriptionResult(
            text=full_text,
            segments=result_segments,
            language=info.language,
            language_probability=info.language_probability,
            duration_audio=info.duration,
            duration_processing=elapsed,
            model_size=self._model_size,
            device=self._device,
        )

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        config: Optional[TranscriptionConfig] = None,
        suffix: str = ".wav",
    ) -> TranscriptionResult:
        """Transcribe audio from raw bytes.

        Writes to a temp file, transcribes, then cleans up.
        Used by the API endpoint when receiving uploaded blobs.
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.transcribe(tmp_path, config)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def detect_language(
        self,
        audio_path: str,
    ) -> Tuple[str, float]:
        """Detect the language of an audio file.

        Returns:
            (language_code, probability) e.g. ("en", 0.98)
        """
        model = self._ensure_model()
        _, info = model.transcribe(audio_path, beam_size=1)
        return info.language, info.language_probability

    @staticmethod
    def list_models() -> List[STTModelInfo]:
        """List all available STT models."""
        return AVAILABLE_MODELS

    def get_info(self) -> dict:
        """Get current transcriber state info."""
        return {
            "model_size": self._model_size,
            "device": self._device,
            "compute_type": self._compute_type,
            "loaded": self._loaded,
            "has_faster_whisper": HAS_FASTER_WHISPER,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_default_transcriber: Optional[VoxTranscriber] = None


def get_transcriber(model_size: str = "base") -> VoxTranscriber:
    """Get or create the default transcriber."""
    global _default_transcriber
    if _default_transcriber is None:
        _default_transcriber = VoxTranscriber(model_size=model_size)
    return _default_transcriber


def transcribe(
    audio_path: str,
    config: Optional[TranscriptionConfig] = None,
) -> TranscriptionResult:
    """Quick transcription using the default transcriber."""
    return get_transcriber().transcribe(audio_path, config)
