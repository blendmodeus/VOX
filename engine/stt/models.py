"""
VØX STT Models
--------------

Data models for the governed Speech-to-Text pipeline.
Mirrors the TTS synthesis models pattern.

TranscriptionConfig → VoxTranscriber → TranscriptionResult
                                              ↓
                                        [STTGovernor]
                                              ↓
                                   GovernedTranscriptionResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class STTModelSize(str, Enum):
    """Available faster-whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V3 = "large-v3"
    DISTIL_LARGE_V3 = "distil-large-v3"


class STTAudioFormat(str, Enum):
    """Supported input audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    WEBM = "webm"
    M4A = "m4a"


class STTDevice(str, Enum):
    """Compute device."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


@dataclass
class TranscriptionConfig:
    """Configuration for a transcription request.

    Mirrors the frontend MODEL/LANG buttons on the KITT overlay.
    """
    model_size: STTModelSize = STTModelSize.BASE
    language: Optional[str] = None  # None = auto-detect
    beam_size: int = 5
    word_timestamps: bool = True
    vad_filter: bool = True
    vad_threshold: float = 0.5
    min_silence_duration_ms: int = 500
    condition_on_previous_text: bool = True
    temperature: float = 0.0
    govern: bool = True  # Apply STTGovernor post-processing

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_size": self.model_size.value,
            "language": self.language,
            "beam_size": self.beam_size,
            "word_timestamps": self.word_timestamps,
            "vad_filter": self.vad_filter,
            "vad_threshold": self.vad_threshold,
            "govern": self.govern,
        }


@dataclass
class WordTimestamp:
    """Word-level timing for precise alignment."""
    word: str
    start: float  # seconds
    end: float    # seconds
    probability: float  # 0.0-1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "probability": round(self.probability, 3),
        }


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio."""
    id: int
    text: str
    start: float  # seconds
    end: float    # seconds
    confidence: float  # avg_logprob mapped to 0-1
    words: List[WordTimestamp] = field(default_factory=list)
    no_speech_prob: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "confidence": round(self.confidence, 3),
            "no_speech_prob": round(self.no_speech_prob, 3),
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class TranscriptionResult:
    """Complete result from transcription.

    This is the raw result BEFORE governance.
    """
    text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: str = "en"
    language_probability: float = 1.0
    duration_audio: float = 0.0  # audio length in seconds
    duration_processing: float = 0.0  # time to transcribe
    model_size: str = "base"
    device: str = "cpu"

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0

    @property
    def avg_confidence(self) -> float:
        if not self.segments:
            return 0.0
        return sum(s.confidence for s in self.segments) / len(self.segments)

    @property
    def rtf(self) -> float:
        """Real-time factor: processing time / audio duration."""
        if self.duration_audio == 0:
            return 0.0
        return self.duration_processing / self.duration_audio

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": round(self.language_probability, 3),
            "duration_audio": round(self.duration_audio, 3),
            "duration_processing": round(self.duration_processing, 3),
            "rtf": round(self.rtf, 3),
            "word_count": self.word_count,
            "avg_confidence": round(self.avg_confidence, 3),
            "model_size": self.model_size,
            "device": self.device,
            "segments": [s.to_dict() for s in self.segments],
            "status": "success",
        }


@dataclass
class STTModelInfo:
    """Metadata about an available STT model."""
    name: str
    size: STTModelSize
    parameters: str  # e.g. "39M", "1.55B"
    size_mb: int
    description: str
    multilingual: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size.value,
            "parameters": self.parameters,
            "size_mb": self.size_mb,
            "description": self.description,
            "multilingual": self.multilingual,
        }


# ============================================================================
# MODEL CATALOG
# ============================================================================

AVAILABLE_MODELS: List[STTModelInfo] = [
    STTModelInfo(
        name="tiny",
        size=STTModelSize.TINY,
        parameters="39M",
        size_mb=75,
        description="Fastest — good for quick dictation",
        multilingual=True,
    ),
    STTModelInfo(
        name="base",
        size=STTModelSize.BASE,
        parameters="74M",
        size_mb=140,
        description="Balanced speed and accuracy (default)",
        multilingual=True,
    ),
    STTModelInfo(
        name="small",
        size=STTModelSize.SMALL,
        parameters="244M",
        size_mb=460,
        description="High accuracy for most use cases",
        multilingual=True,
    ),
    STTModelInfo(
        name="medium",
        size=STTModelSize.MEDIUM,
        parameters="769M",
        size_mb=1500,
        description="Near-SOTA accuracy, slower inference",
        multilingual=True,
    ),
    STTModelInfo(
        name="large-v3",
        size=STTModelSize.LARGE_V3,
        parameters="1.55B",
        size_mb=3000,
        description="Best quality — production transcription",
        multilingual=True,
    ),
    STTModelInfo(
        name="distil-large-v3",
        size=STTModelSize.DISTIL_LARGE_V3,
        parameters="756M",
        size_mb=1500,
        description="Distilled large — fast CUDA inference",
        multilingual=False,
    ),
]


def get_model_info(size: str) -> Optional[STTModelInfo]:
    """Get model info by size string."""
    for m in AVAILABLE_MODELS:
        if m.size.value == size or m.name == size:
            return m
    return None
