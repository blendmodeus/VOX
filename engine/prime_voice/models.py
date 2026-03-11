"""
PRIME Voice Models
------------------

Data models for PRIME's voice identity, speaking modes, and utterance output.

PRIME's voice is defined by three layers:
1. Identity: The immutable vocal DNA (8-dim vector, genome profile, biometric anchor)
2. Speaking Mode: Context-adaptive delivery (briefing, conversational, alert, reflective)
3. Utterance: The rendered speech output with full provenance
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Enums
# =============================================================================

class SpeakingModeType(Enum):
    """PRIME's speaking modes - how it delivers information based on context."""
    BRIEFING = "briefing"              # Status reports, summaries - clear, efficient
    CONVERSATIONAL = "conversational"  # Dialogue, Q&A - warm, engaging
    ALERT = "alert"                    # Warnings, urgent info - firm, immediate
    REFLECTIVE = "reflective"          # Analysis, reasoning - measured, thoughtful
    DIRECTIVE = "directive"            # Commands, instructions - authoritative, precise
    EMPATHETIC = "empathetic"          # Support, acknowledgment - warm, understanding
    CEREMONIAL = "ceremonial"          # Announcements, milestones - elevated, resonant


class PrimeVoiceState(Enum):
    """Current state of PRIME's voice system."""
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    SPEAKING = "speaking"
    STREAMING = "streaming"
    PAUSED = "paused"
    ERROR = "error"


class IdentityLockLevel(Enum):
    """How strictly the voice identity is enforced."""
    LOCKED = "locked"          # No deviation allowed - production
    CALIBRATING = "calibrating"  # Minor adjustments permitted - tuning phase
    OPEN = "open"              # Full modification allowed - development only


class UtteranceType(Enum):
    """Classification of what PRIME is saying."""
    STATUS = "status"              # System status report
    RESPONSE = "response"          # Answer to a question
    ALERT = "alert"                # Warning or urgent notification
    GREETING = "greeting"          # Session start/end
    NARRATION = "narration"        # Explaining a process
    COMMAND_ECHO = "command_echo"  # Confirming a command
    REFLECTION = "reflection"      # Analysis or reasoning output
    ERROR_REPORT = "error_report"  # Error communication


# =============================================================================
# Voice Identity Models
# =============================================================================

@dataclass
class PrimeVocalDNA:
    """
    PRIME's immutable vocal characteristics - the voice genome definition.

    These values define what PRIME *sounds like* at the deepest level.
    Derived from the Voice Genome sociometric/biometric targets.
    """
    # Biometric targets (what the voice physically sounds like)
    target_pitch_hz: float = 115.0          # Low-mid male range (Jarvis-like)
    target_pitch_variance: float = 0.3      # Controlled, not monotone
    target_speaking_rate: float = 0.95      # Slightly measured, not rushed
    target_resonance: float = 0.8           # Rich chest voice
    target_breathiness: float = 0.1         # Clean, clear signal
    target_vocal_fry: float = 0.05          # Minimal - precision signal

    # Sociometric targets (how the voice affects listeners)
    target_authority: float = 0.85          # High authority without intimidation
    target_warmth: float = 0.6             # Warm but not overly familiar
    target_trust: float = 0.9              # Maximum trustworthiness
    target_charisma: float = 0.75          # Engaging without showmanship
    target_approachability: float = 0.65   # Accessible but professional
    target_credibility: float = 0.95       # Near-maximum expertise signal
    target_dominance: float = 0.7          # Assertive, not commanding

    # Psychometric targets (emotional baseline)
    baseline_calm: float = 0.85            # Stress-immune baseline
    baseline_confidence: float = 0.9       # Unwavering confidence
    baseline_focus: float = 0.95           # Laser precision
    emotional_range: float = 0.4           # Controlled emotional expression

    def to_dict(self) -> Dict[str, float]:
        """Serialize vocal DNA to dictionary."""
        return {
            "target_pitch_hz": self.target_pitch_hz,
            "target_pitch_variance": self.target_pitch_variance,
            "target_speaking_rate": self.target_speaking_rate,
            "target_resonance": self.target_resonance,
            "target_breathiness": self.target_breathiness,
            "target_vocal_fry": self.target_vocal_fry,
            "target_authority": self.target_authority,
            "target_warmth": self.target_warmth,
            "target_trust": self.target_trust,
            "target_charisma": self.target_charisma,
            "target_approachability": self.target_approachability,
            "target_credibility": self.target_credibility,
            "target_dominance": self.target_dominance,
            "baseline_calm": self.baseline_calm,
            "baseline_confidence": self.baseline_confidence,
            "baseline_focus": self.baseline_focus,
            "emotional_range": self.emotional_range,
        }


@dataclass
class PrimeVoiceVector:
    """
    PRIME's position in the 8-dimensional voice space.

    Maps directly to VoiceVector from axiom_vox.voice_space.
    These values are LOCKED once calibrated - PRIME always sounds like PRIME.
    """
    formality: float = 0.6        # Formal but not stiff
    temperature: float = 0.4      # Warm side of professional
    energy: float = -0.2          # Calm, measured energy
    authority: float = 0.7        # Clearly authoritative
    abstraction: float = 0.3      # Concrete when possible, abstract when needed
    intimacy: float = -0.1        # Professional distance, not cold
    certainty: float = 0.6        # Confident, acknowledges uncertainty
    complexity: float = 0.4       # Adapts complexity to audience

    def to_dict(self) -> Dict[str, float]:
        """Serialize to dictionary."""
        return {
            "formality": self.formality,
            "temperature": self.temperature,
            "energy": self.energy,
            "authority": self.authority,
            "abstraction": self.abstraction,
            "intimacy": self.intimacy,
            "certainty": self.certainty,
            "complexity": self.complexity,
        }

    def to_list(self) -> List[float]:
        """Convert to 8-dim list for VoiceVector compatibility."""
        return [
            self.formality,
            self.temperature,
            self.energy,
            self.authority,
            self.abstraction,
            self.intimacy,
            self.certainty,
            self.complexity,
        ]


@dataclass
class PrimeVoiceIdentity:
    """
    The complete, locked identity of PRIME's voice.

    This is PRIME's vocal fingerprint - combining the 8D voice vector,
    vocal DNA targets, biometric anchor, and identity lock.
    """
    # Identity
    identity_id: str = "PRIME_VOICE_001"
    identity_name: str = "PRIME"
    identity_version: str = "1.0.0"
    lock_level: IdentityLockLevel = IdentityLockLevel.LOCKED

    # Voice definition layers
    voice_vector: PrimeVoiceVector = field(default_factory=PrimeVoiceVector)
    vocal_dna: PrimeVocalDNA = field(default_factory=PrimeVocalDNA)

    # Biometric anchor (populated after enrollment)
    biometric_embedding: Optional[List[float]] = None
    biometric_enrolled: bool = False
    enrollment_timestamp: Optional[datetime] = None

    # Voice ID in the VØX system
    vox_voice_id: str = "prime_sovereign"
    emotion_preset: str = "professional"

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    description: str = "PRIME Sovereign Voice - All Signal, ZERO Noise"

    def is_locked(self) -> bool:
        """Check if identity modifications are blocked."""
        return self.lock_level == IdentityLockLevel.LOCKED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize identity to dictionary."""
        return {
            "identity_id": self.identity_id,
            "identity_name": self.identity_name,
            "identity_version": self.identity_version,
            "lock_level": self.lock_level.value,
            "voice_vector": self.voice_vector.to_dict(),
            "vocal_dna": self.vocal_dna.to_dict(),
            "biometric_enrolled": self.biometric_enrolled,
            "vox_voice_id": self.vox_voice_id,
            "emotion_preset": self.emotion_preset,
            "description": self.description,
        }


# =============================================================================
# Speaking Mode Models
# =============================================================================

@dataclass
class SpeakingModeProfile:
    """
    Defines how a speaking mode modifies PRIME's base voice.

    Each mode applies deltas to the base voice vector and prosody targets.
    The base identity remains locked - modes only adjust delivery.
    """
    mode: SpeakingModeType
    name: str
    description: str

    # Voice vector deltas (applied on top of base PrimeVoiceVector)
    formality_delta: float = 0.0
    temperature_delta: float = 0.0
    energy_delta: float = 0.0
    authority_delta: float = 0.0
    certainty_delta: float = 0.0
    intimacy_delta: float = 0.0

    # Prosody adjustments
    rate_multiplier: float = 1.0      # Speaking rate modifier
    pitch_shift: float = 0.0          # Semitones from base
    pitch_variance_delta: float = 0.0  # More/less expressive
    pause_multiplier: float = 1.0     # Pause duration modifier

    # Emotional coloring (overrides if set)
    warmth_override: Optional[float] = None
    confidence_override: Optional[float] = None
    energy_override: Optional[float] = None

    # Emphasis behavior
    emphasis_strength: float = 0.5     # How strongly to emphasize key words
    sentence_boundary_pause: float = 0.3  # Seconds between sentences

    def to_dict(self) -> Dict[str, Any]:
        """Serialize mode profile."""
        return {
            "mode": self.mode.value,
            "name": self.name,
            "description": self.description,
            "rate_multiplier": self.rate_multiplier,
            "pitch_shift": self.pitch_shift,
            "pause_multiplier": self.pause_multiplier,
            "emphasis_strength": self.emphasis_strength,
        }


@dataclass
class ModeTransition:
    """Describes a transition between speaking modes."""
    from_mode: SpeakingModeType
    to_mode: SpeakingModeType
    crossfade_seconds: float = 0.5     # Audio crossfade duration
    pause_between: float = 0.2          # Silence between modes
    gradual: bool = True                # Gradual or instant switch


# =============================================================================
# Utterance Models
# =============================================================================

@dataclass
class PrimeUtterance:
    """
    A single rendered speech output from PRIME.

    Contains the audio, metadata, provenance, and analysis of what PRIME said.
    Every utterance is traceable back to its source text, mode, and identity.
    """
    # Identification
    utterance_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Source
    source_text: str = ""
    utterance_type: UtteranceType = UtteranceType.RESPONSE
    speaking_mode: SpeakingModeType = SpeakingModeType.CONVERSATIONAL

    # Audio output
    audio_data: Optional[bytes] = None
    audio_path: Optional[str] = None
    duration_seconds: float = 0.0
    sample_rate: int = 24000
    audio_format: str = "wav"

    # Voice parameters used
    voice_id: str = "prime_sovereign"
    voice_vector_used: Optional[Dict[str, float]] = None
    prosody_applied: Optional[Dict[str, Any]] = None
    emotion_preset_used: str = "professional"

    # Governance
    governance_passed: bool = True
    governance_report: Optional[Dict[str, Any]] = None

    # Identity verification
    identity_verified: bool = False
    identity_similarity: float = 0.0

    # Quality
    audio_quality_score: float = 0.0
    naturalness_score: float = 0.0

    # Context
    session_id: Optional[str] = None
    request_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize utterance metadata (without audio bytes)."""
        return {
            "utterance_id": self.utterance_id,
            "timestamp": self.timestamp.isoformat(),
            "source_text": self.source_text,
            "utterance_type": self.utterance_type.value,
            "speaking_mode": self.speaking_mode.value,
            "duration_seconds": self.duration_seconds,
            "voice_id": self.voice_id,
            "emotion_preset_used": self.emotion_preset_used,
            "governance_passed": self.governance_passed,
            "identity_verified": self.identity_verified,
            "identity_similarity": self.identity_similarity,
            "audio_quality_score": self.audio_quality_score,
            "naturalness_score": self.naturalness_score,
            "session_id": self.session_id,
        }


@dataclass
class PrimeVoiceSession:
    """
    A voice session tracks PRIME's speaking activity over time.

    Maintains state across multiple utterances for consistent delivery.
    """
    session_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    current_mode: SpeakingModeType = SpeakingModeType.CONVERSATIONAL
    state: PrimeVoiceState = PrimeVoiceState.UNINITIALIZED

    # Session metrics
    utterance_count: int = 0
    total_duration_seconds: float = 0.0
    mode_switches: int = 0

    # History (last N utterances for context)
    recent_utterance_ids: List[str] = field(default_factory=list)
    mode_history: List[SpeakingModeType] = field(default_factory=list)

    # Identity consistency
    identity_checks_passed: int = 0
    identity_checks_failed: int = 0
    average_identity_similarity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session state."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "current_mode": self.current_mode.value,
            "state": self.state.value,
            "utterance_count": self.utterance_count,
            "total_duration_seconds": self.total_duration_seconds,
            "mode_switches": self.mode_switches,
            "average_identity_similarity": self.average_identity_similarity,
        }


@dataclass
class PrimeVoiceConfig:
    """
    Configuration for the PRIME Voice system.
    """
    # Identity
    identity: PrimeVoiceIdentity = field(default_factory=PrimeVoiceIdentity)

    # Synthesis settings
    model_size: str = "small"              # "small" (0.6B) or "large" (1.7B)
    sample_rate: int = 24000
    audio_format: str = "wav"

    # Streaming
    enable_streaming: bool = True
    stream_chunk_size: int = 4096
    max_first_chunk_ms: int = 500

    # Identity enforcement
    verify_identity_every_n: int = 10      # Verify biometric every N utterances
    identity_similarity_threshold: float = 0.75
    enforce_governance: bool = True

    # Mode behavior
    default_mode: SpeakingModeType = SpeakingModeType.CONVERSATIONAL
    auto_detect_mode: bool = True          # Auto-detect mode from context
    mode_transition_pause: float = 0.3     # Seconds between mode switches

    # Logging
    log_utterances: bool = True
    max_session_history: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config."""
        return {
            "identity": self.identity.to_dict(),
            "model_size": self.model_size,
            "sample_rate": self.sample_rate,
            "enable_streaming": self.enable_streaming,
            "verify_identity_every_n": self.verify_identity_every_n,
            "default_mode": self.default_mode.value,
            "auto_detect_mode": self.auto_detect_mode,
            "enforce_governance": self.enforce_governance,
        }
