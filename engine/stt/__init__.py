"""
VØX STT Module
--------------

Governed Speech-to-Text pipeline for AXIØM VØX.
Replaces Glaido/Whisper with self-hosted, governed transcription.

Components:
    - VoxTranscriber: Core transcription engine (faster-whisper)
    - STTGovernor: Post-transcription governance (PII, content, laws)
    - TextFormatter: AI text formatting (filler removal, LLM polish)
    - StreamingTranscriber: Real-time streaming transcription
    - Models: TranscriptionConfig, TranscriptionResult, etc.

Quick Start:
    from axiom_vox.stt import transcribe, govern_transcription

    result = transcribe("audio.wav")
    governed = govern_transcription(result)
    print(governed.governed_text)

Full Control:
    from axiom_vox.stt import VoxTranscriber, STTGovernor, TranscriptionConfig

    transcriber = VoxTranscriber(model_size="small")
    config = TranscriptionConfig(language="en", word_timestamps=True)
    result = transcriber.transcribe("audio.wav", config)

    governor = STTGovernor()
    governed = governor.govern(result)
"""

# Models
from axiom_vox.stt.models import (
    STTModelSize,
    STTAudioFormat,
    STTDevice,
    TranscriptionConfig,
    WordTimestamp,
    TranscriptionSegment,
    TranscriptionResult,
    STTModelInfo,
    AVAILABLE_MODELS,
    get_model_info,
)

# Transcriber
from axiom_vox.stt.transcriber import (
    VoxTranscriber,
    get_transcriber,
    transcribe,
    detect_device,
    HAS_FASTER_WHISPER,
)

# Governor
from axiom_vox.stt.governor import (
    STTGovernor,
    STTGovernanceConfig,
    STTGovernanceResult,
    PIIRedaction,
    RedactionType,
    ContentFlag,
    get_stt_governor,
    govern_transcription,
)

# Streaming
from axiom_vox.stt.streaming import (
    StreamingTranscriber,
    STTStreamConfig,
    STTStreamSession,
    STTStreamState,
    STTStreamMessage,
    PartialTranscript,
    get_streaming_transcriber,
)

# Formatter
from axiom_vox.stt.formatter import (
    TextFormatter,
    FormatMode,
    FormatResult,
    get_formatter,
    clean_text,
)

# Hotwords (Custom Dictionary)
from axiom_vox.stt.hotwords import (
    HotwordManager,
    HotwordEntry,
    get_hotword_manager,
)

# Snippets (Voice Shortcuts)
from axiom_vox.stt.snippets import (
    SnippetManager,
    Snippet,
    get_snippet_manager,
)

# Wake Word Detection
from axiom_vox.stt.wakeword import (
    WakeWordDetector,
    WakeWordConfig,
    get_wake_detector,
)

__all__ = [
    # Models
    "STTModelSize",
    "STTAudioFormat",
    "STTDevice",
    "TranscriptionConfig",
    "WordTimestamp",
    "TranscriptionSegment",
    "TranscriptionResult",
    "STTModelInfo",
    "AVAILABLE_MODELS",
    "get_model_info",
    # Transcriber
    "VoxTranscriber",
    "get_transcriber",
    "transcribe",
    "detect_device",
    "HAS_FASTER_WHISPER",
    # Governor
    "STTGovernor",
    "STTGovernanceConfig",
    "STTGovernanceResult",
    "PIIRedaction",
    "RedactionType",
    "ContentFlag",
    "get_stt_governor",
    "govern_transcription",
    # Streaming
    "StreamingTranscriber",
    "STTStreamConfig",
    "STTStreamSession",
    "STTStreamState",
    "STTStreamMessage",
    "PartialTranscript",
    "get_streaming_transcriber",
    # Formatter
    "TextFormatter",
    "FormatMode",
    "FormatResult",
    "get_formatter",
    "clean_text",
    # Hotwords
    "HotwordManager",
    "HotwordEntry",
    "get_hotword_manager",
    # Snippets
    "SnippetManager",
    "Snippet",
    "get_snippet_manager",
    # Wake Word
    "WakeWordDetector",
    "WakeWordConfig",
    "get_wake_detector",
]
