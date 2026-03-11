"""
Unified Voice Pipeline Models
-----------------------------

Dataclasses for the unified VØX voice pipeline.

AXIØM Phase 6: System - "Integrate the parts"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Union
import time
import uuid


class PipelineStage(str, Enum):
    """Stages in the unified pipeline."""
    INTAKE = "intake"  # Initial request processing
    IDENTITY = "identity"  # Biometric identification/verification
    CONSENT = "consent"  # Consent verification
    GOVERNANCE = "governance"  # AXIOM governance check
    ROUTING = "routing"  # Voice selection/routing
    SYNTHESIS = "synthesis"  # Audio generation
    QUALITY = "quality"  # Quality assessment
    DELIVERY = "delivery"  # Final output


class PipelineStatus(str, Enum):
    """Overall pipeline status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Blocked by governance/consent


class VoiceRouteType(str, Enum):
    """How voice was selected."""
    EXPLICIT = "explicit"  # User specified voice_id
    BIOMETRIC = "biometric"  # Matched via speaker embedding
    CONTEXT = "context"  # Selected based on content analysis
    DEFAULT = "default"  # Fallback to default voice
    CLONED = "cloned"  # Using cloned voice


class ConsentScope(str, Enum):
    """Scope of consent."""
    SYNTHESIS = "synthesis"  # Basic TTS
    CLONING = "cloning"  # Voice cloning
    BIOMETRIC = "biometric"  # Biometric enrollment
    STREAMING = "streaming"  # Real-time streaming
    COMMERCIAL = "commercial"  # Commercial use
    THIRD_PARTY = "third_party"  # Third-party access


class QualityGate(str, Enum):
    """Quality gate status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineRequest:
    """Request to the unified pipeline."""
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")

    # Input
    text: Optional[str] = None
    ssml: Optional[str] = None
    audio_input: Optional[bytes] = None  # For voice identification

    # Voice selection
    voice_id: Optional[str] = None
    speaker_embedding: Optional[bytes] = None  # For biometric routing
    character_name: Optional[str] = None  # For multi-voice

    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Options
    require_biometric_verification: bool = False
    require_consent_check: bool = True
    enable_quality_monitoring: bool = True
    stream: bool = False

    # Emotion/prosody
    emotion_preset: Optional[str] = None
    speaking_rate: float = 1.0

    # Timestamps
    created_at: float = field(default_factory=time.time)


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    stage: PipelineStage
    success: bool
    duration_ms: float = 0.0
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class IdentityResult:
    """Result of identity verification stage."""
    identified: bool = False
    verified: bool = False
    voice_id: Optional[str] = None
    similarity_score: float = 0.0
    liveness_passed: bool = False
    liveness_score: float = 0.0
    speaker_embedding: Optional[bytes] = None
    message: str = ""


@dataclass
class ConsentResult:
    """Result of consent verification stage."""
    granted: bool = False
    scopes: List[ConsentScope] = field(default_factory=list)
    consent_id: Optional[int] = None
    expires_at: Optional[float] = None
    restrictions: List[str] = field(default_factory=list)
    message: str = ""


@dataclass
class GovernanceResult:
    """Result of governance stage."""
    approved: bool = False
    action: str = "allow"  # allow, repair, refuse
    original_text: str = ""
    governed_text: str = ""
    repairs_made: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RouteResult:
    """Result of voice routing stage."""
    voice_id: str = ""
    route_type: VoiceRouteType = VoiceRouteType.DEFAULT
    voice_verified: bool = False
    voice_quality_score: float = 0.0
    adapter_path: Optional[str] = None  # For cloned voices
    voice_config: Dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False
    message: str = ""


@dataclass
class SynthesisResult:
    """Result of synthesis stage."""
    audio_data: Optional[bytes] = None
    audio_format: str = "wav"
    duration_seconds: float = 0.0
    sample_rate: int = 24000
    synthesis_time_ms: float = 0.0
    rtf: float = 0.0  # Real-time factor
    chunks_generated: int = 0
    message: str = ""


@dataclass
class QualityResult:
    """Result of quality assessment stage."""
    overall_score: float = 0.0
    gate_status: QualityGate = QualityGate.SKIPPED

    # Component scores
    snr_db: float = 0.0
    naturalness_score: float = 0.0
    intelligibility_score: float = 0.0
    consistency_score: float = 0.0

    # Issues detected
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Thresholds
    min_acceptable_score: float = 0.6
    target_score: float = 0.8


@dataclass
class PipelineResponse:
    """Complete response from unified pipeline."""
    request_id: str
    status: PipelineStatus = PipelineStatus.PENDING

    # Stage results
    stages_completed: List[PipelineStage] = field(default_factory=list)
    stage_results: Dict[str, StageResult] = field(default_factory=dict)

    # Key results
    identity: Optional[IdentityResult] = None
    consent: Optional[ConsentResult] = None
    governance: Optional[GovernanceResult] = None
    route: Optional[RouteResult] = None
    synthesis: Optional[SynthesisResult] = None
    quality: Optional[QualityResult] = None

    # Output
    audio_data: Optional[bytes] = None
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None

    # Metadata
    total_duration_ms: float = 0.0
    voice_id_used: Optional[str] = None
    governed_text: Optional[str] = None

    # Errors/warnings
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Timestamps
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class VoiceProfile:
    """Unified voice profile combining all voice data."""
    voice_id: str
    name: str = ""

    # Status
    is_active: bool = True
    is_cloned: bool = False
    is_biometric_enrolled: bool = False

    # Biometric data
    biometric_template_id: Optional[str] = None
    biometric_confidence: float = 0.0
    last_verified_at: Optional[float] = None

    # Clone data
    adapter_id: Optional[str] = None
    adapter_path: Optional[str] = None
    clone_quality_score: float = 0.0

    # Consent
    consent_scopes: List[ConsentScope] = field(default_factory=list)
    consent_expires_at: Optional[float] = None

    # Voice characteristics
    voice_vector: Optional[bytes] = None  # 8-dim voice space
    default_emotion: str = "neutral"
    languages: List[str] = field(default_factory=lambda: ["en"])

    # Usage
    total_synthesis_count: int = 0
    last_used_at: Optional[float] = None

    # Metadata
    owner_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for unified pipeline."""
    # Identity
    enable_biometric_identification: bool = True
    require_biometric_for_clones: bool = True
    biometric_similarity_threshold: float = 0.75

    # Consent
    require_consent: bool = True
    default_consent_scopes: List[ConsentScope] = field(
        default_factory=lambda: [ConsentScope.SYNTHESIS]
    )

    # Governance
    strict_governance: bool = False
    allow_repairs: bool = True

    # Routing
    enable_context_routing: bool = True
    default_voice_id: str = "axiom_default"
    fallback_voice_id: str = "axiom_fallback"

    # Quality
    enable_quality_gates: bool = True
    min_quality_score: float = 0.6
    block_on_quality_failure: bool = False

    # Performance
    max_text_length: int = 5000
    synthesis_timeout_seconds: float = 30.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # Streaming
    chunk_size_ms: int = 100
    enable_streaming_quality: bool = True


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""
    request_id: str
    timestamp: float = field(default_factory=time.time)

    # Timing
    total_duration_ms: float = 0.0
    stage_durations: Dict[str, float] = field(default_factory=dict)

    # Identity
    biometric_checked: bool = False
    biometric_verified: bool = False
    identity_latency_ms: float = 0.0

    # Consent
    consent_checked: bool = False
    consent_granted: bool = False

    # Routing
    route_type: str = ""
    voice_id: str = ""
    fallback_used: bool = False

    # Synthesis
    text_length: int = 0
    audio_duration_seconds: float = 0.0
    synthesis_latency_ms: float = 0.0
    rtf: float = 0.0

    # Quality
    quality_score: float = 0.0
    quality_gate_passed: bool = True

    # Status
    success: bool = False
    error_stage: Optional[str] = None
    error_message: Optional[str] = None


# Constants
DEFAULT_PIPELINE_CONFIG = PipelineConfig()

QUALITY_THRESHOLDS = {
    "excellent": 0.9,
    "good": 0.8,
    "acceptable": 0.6,
    "poor": 0.0,
}

STAGE_ORDER = [
    PipelineStage.INTAKE,
    PipelineStage.IDENTITY,
    PipelineStage.CONSENT,
    PipelineStage.GOVERNANCE,
    PipelineStage.ROUTING,
    PipelineStage.SYNTHESIS,
    PipelineStage.QUALITY,
    PipelineStage.DELIVERY,
]
