"""
Voice Biometric Models
----------------------

Dataclasses for voice biometric verification.

AXIØM Phase 5: Resonance - "finding signature frequency"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import time


class EmbeddingBackend(str, Enum):
    """Embedding extraction backend."""
    SPECTRAL = "spectral"  # SpectralFingerprint (default, no deps)
    ECAPA = "ecapa"  # ECAPA-TDNN (requires speechbrain)
    WAV2VEC = "wav2vec"  # Wav2Vec2 (requires transformers)


class VerificationStatus(str, Enum):
    """Verification result status."""
    VERIFIED = "verified"
    REJECTED = "rejected"
    LIVENESS_FAILED = "liveness_failed"
    NOT_ENROLLED = "not_enrolled"
    TEMPLATE_REVOKED = "template_revoked"
    ERROR = "error"


class EnrollmentStatus(str, Enum):
    """Enrollment result status."""
    SUCCESS = "success"
    ALREADY_ENROLLED = "already_enrolled"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    LOW_QUALITY = "low_quality"
    CONSENT_REQUIRED = "consent_required"
    ERROR = "error"


class LivenessStatus(str, Enum):
    """Liveness check status."""
    PASSED = "passed"
    FAILED = "failed"
    REPLAY_DETECTED = "replay_detected"
    DEEPFAKE_DETECTED = "deepfake_detected"
    INSUFFICIENT_AUDIO = "insufficient_audio"
    ERROR = "error"


class DriftSeverity(str, Enum):
    """Voice drift severity level."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    ANOMALOUS = "anomalous"


class BiometricAction(str, Enum):
    """Actions for audit logging."""
    ENROLL = "enroll"
    VERIFY = "verify"
    UPDATE = "update"
    REVOKE = "revoke"
    DRIFT_CHECK = "drift_check"


@dataclass
class BiometricTemplate:
    """Enrolled voice biometric template."""
    template_id: str
    voice_id: str
    embedding: bytes  # Serialized numpy array
    embedding_version: str = "1.0"
    embedding_backend: EmbeddingBackend = EmbeddingBackend.SPECTRAL
    embedding_dim: int = 256
    sample_count: int = 1
    confidence: float = 0.0
    consent_id: Optional[int] = None
    owner_id: Optional[str] = None
    enrolled_at: float = field(default_factory=time.time)
    updated_at: Optional[float] = None
    is_active: bool = True
    revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrollmentResult:
    """Result of voice enrollment."""
    status: EnrollmentStatus
    template_id: Optional[str] = None
    voice_id: Optional[str] = None
    embedding_dim: int = 256
    sample_count: int = 0
    confidence: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    average_quality: float = 0.0
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    enrolled_at: Optional[float] = None


@dataclass
class VerificationResult:
    """Result of voice verification."""
    status: VerificationStatus
    voice_id: str
    similarity_score: float = 0.0
    threshold: float = 0.75
    is_verified: bool = False
    liveness_passed: bool = False
    liveness_score: float = 0.0
    liveness_details: Dict[str, Any] = field(default_factory=dict)
    drift_detected: bool = False
    drift_severity: DriftSeverity = DriftSeverity.NONE
    confidence: float = 0.0
    message: str = ""
    verified_at: float = field(default_factory=time.time)


@dataclass
class LivenessResult:
    """Result of liveness detection."""
    status: LivenessStatus
    overall_score: float = 0.0
    passed: bool = False

    # Individual check scores
    replay_score: float = 0.0  # Higher = more likely real
    deepfake_score: float = 0.0  # Higher = more likely real
    breath_score: float = 0.0  # Natural breath/pop detection
    channel_score: float = 0.0  # Recording environment analysis
    prosody_score: float = 0.0  # Natural timing patterns

    # Detection flags
    replay_detected: bool = False
    deepfake_detected: bool = False
    compression_artifacts: bool = False

    # Details
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class DriftReport:
    """Voice drift analysis report."""
    voice_id: str
    severity: DriftSeverity = DriftSeverity.NONE

    # Drift measurements
    short_term_drift: float = 0.0  # Session-to-session
    long_term_drift: float = 0.0  # Over months/years
    trend_direction: str = "stable"  # "stable", "increasing", "decreasing"

    # Thresholds
    short_term_threshold: float = 0.15
    long_term_threshold: float = 0.25

    # Recommendations
    requires_re_enrollment: bool = False
    update_recommended: bool = False

    # History
    sample_count: int = 0
    first_sample_date: Optional[float] = None
    last_sample_date: Optional[float] = None

    # Analysis
    anomaly_detected: bool = False
    anomaly_details: Optional[str] = None

    message: str = ""
    analyzed_at: float = field(default_factory=time.time)


@dataclass
class EmbeddingResult:
    """Result of embedding extraction."""
    embedding: bytes  # Serialized numpy array
    embedding_dim: int
    backend: EmbeddingBackend
    quality_score: float = 0.0
    duration_seconds: float = 0.0
    sample_rate: int = 24000
    is_valid: bool = True
    message: str = ""


@dataclass
class BiometricAuditEntry:
    """Audit log entry for biometric operations."""
    timestamp: float
    voice_id: str
    action: BiometricAction
    result: str  # "success", "failure", "blocked"
    similarity_score: Optional[float] = None
    liveness_score: Optional[float] = None
    liveness_passed: Optional[bool] = None
    drift_detected: bool = False
    error_message: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BiometricConfig:
    """Configuration for biometric service."""
    # Embedding
    embedding_backend: EmbeddingBackend = EmbeddingBackend.SPECTRAL
    embedding_dim: int = 256

    # Thresholds
    similarity_threshold: float = 0.75
    liveness_threshold: float = 0.80
    short_term_drift_threshold: float = 0.15
    long_term_drift_threshold: float = 0.25

    # Enrollment
    min_enrollment_samples: int = 3
    max_enrollment_samples: int = 10
    min_sample_duration: float = 3.0  # seconds
    max_sample_duration: float = 30.0  # seconds

    # Liveness
    require_liveness: bool = True
    liveness_checks: List[str] = field(default_factory=lambda: [
        "replay", "deepfake", "breath", "channel", "prosody"
    ])

    # Privacy
    encrypt_embeddings: bool = True
    audit_all_operations: bool = True
    retention_days: int = 365

    # Performance
    cache_embeddings: bool = True
    cache_ttl_seconds: int = 3600


# Constants
EMBEDDING_VERSION = "1.0"
SPECTRAL_EMBEDDING_DIM = 256
ECAPA_EMBEDDING_DIM = 192
WAV2VEC_EMBEDDING_DIM = 768

# Thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_LIVENESS_THRESHOLD = 0.80
DRIFT_SHORT_TERM_THRESHOLD = 0.15
DRIFT_LONG_TERM_THRESHOLD = 0.25
