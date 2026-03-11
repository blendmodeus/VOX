"""
VØX PRIME Voice
---------------

PRIME's sovereign voice - the voice of AXIOM intelligence.

Like Jarvis to Tony Stark, PRIME Voice gives the AXIOM system
a consistent, recognizable, and trustworthy vocal identity.

PRIME's voice is defined by three layers:
1. **Identity**: Immutable vocal DNA (8-dim vector + genome profile + biometric anchor)
2. **Speaking Modes**: Context-aware delivery (briefing, alert, reflective, etc.)
3. **Renderer**: Full pipeline from text to governed, identity-verified speech

Quick Start:
    >>> from axiom_vox.prime_voice import prime_speak, SpeakingModeType
    >>>
    >>> # Simple - PRIME speaks with auto-detected mode
    >>> result = prime_speak("All systems operational. Uptime: 99.9%")
    >>> # -> Renders in BRIEFING mode (auto-detected from metrics)
    >>>
    >>> # Explicit mode
    >>> result = prime_speak("Warning: memory at 95%", mode=SpeakingModeType.ALERT)
    >>>
    >>> # Streaming
    >>> from axiom_vox.prime_voice import prime_stream, StreamEventType
    >>> async for event in prime_stream("Deploying v2.1.0 to production."):
    ...     if event.event_type == StreamEventType.CHUNK:
    ...         play_audio(event.chunk.data)

PRIME's voice: All Signal, ZERO Noise.
"""

from .models import (
    # Enums
    SpeakingModeType,
    PrimeVoiceState,
    IdentityLockLevel,
    UtteranceType,
    # Identity models
    PrimeVocalDNA,
    PrimeVoiceVector,
    PrimeVoiceIdentity,
    # Mode models
    SpeakingModeProfile,
    ModeTransition,
    # Utterance models
    PrimeUtterance,
    PrimeVoiceSession,
    PrimeVoiceConfig,
)

from .identity import (
    IdentityConfig,
    IdentityCheckResult,
    PrimeVoiceIdentityManager,
    get_identity_manager,
    get_prime_identity,
    get_prime_synthesis_params,
)

from .speaking_modes import (
    ModeDetectionResult,
    SpeakingModeManager,
    PRIME_SPEAKING_MODES,
    detect_speaking_mode,
    get_mode_profile,
)

from .renderer import (
    RendererConfig,
    RenderResult,
    PrimeVoiceRenderer,
    get_renderer,
    prime_speak,
)

from .streamer import (
    StreamEventType,
    PrimeAudioChunk,
    PrimeStreamEvent,
    StreamConfig,
    PrimeVoiceStreamer,
    get_streamer,
    prime_stream,
    segment_sentences,
)


__all__ = [
    # Enums
    "SpeakingModeType",
    "PrimeVoiceState",
    "IdentityLockLevel",
    "UtteranceType",
    "StreamEventType",
    # Identity models
    "PrimeVocalDNA",
    "PrimeVoiceVector",
    "PrimeVoiceIdentity",
    # Mode models
    "SpeakingModeProfile",
    "ModeTransition",
    # Utterance models
    "PrimeUtterance",
    "PrimeVoiceSession",
    "PrimeVoiceConfig",
    # Identity manager
    "IdentityConfig",
    "IdentityCheckResult",
    "PrimeVoiceIdentityManager",
    "get_identity_manager",
    "get_prime_identity",
    "get_prime_synthesis_params",
    # Speaking modes
    "ModeDetectionResult",
    "SpeakingModeManager",
    "PRIME_SPEAKING_MODES",
    "detect_speaking_mode",
    "get_mode_profile",
    # Renderer
    "RendererConfig",
    "RenderResult",
    "PrimeVoiceRenderer",
    "get_renderer",
    "prime_speak",
    # Streamer
    "PrimeAudioChunk",
    "PrimeStreamEvent",
    "StreamConfig",
    "PrimeVoiceStreamer",
    "get_streamer",
    "prime_stream",
    "segment_sentences",
]
