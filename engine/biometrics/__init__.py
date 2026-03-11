"""
AXIØM VØX Biometrics
--------------------

Voice biometric verification for identity authentication.

AXIØM Phase 5: Resonance - "finding signature frequency"

v0.10.0: Voice Biometric Verification

Features:
    - SpectralFingerprint: 256-dim speaker embedding (no external deps)
    - LivenessDetector: Anti-spoofing (replay, deepfake detection)
    - DriftMonitor: Voice change detection over time
    - VoiceBiometricService: Complete enrollment/verification workflow

Usage:
    from axiom_vox.biometrics import (
        VoiceBiometricService,
        SpectralFingerprint,
        LivenessDetector,
    )

    # Create service
    service = VoiceBiometricService()

    # Enroll voice (requires consent)
    result = await service.enroll(
        voice_id="user_123",
        audio_samples=[sample1, sample2, sample3],
        owner_id="owner_456",
        consent_token="token_xyz",
    )

    # Verify voice
    result = await service.verify(
        voice_id="user_123",
        audio_sample=audio_bytes,
        require_liveness=True,
    )

    # Check drift
    drift = await service.check_drift("user_123")
"""

# Models
from .models import (
    # Enums
    EmbeddingBackend,
    VerificationStatus,
    EnrollmentStatus,
    LivenessStatus,
    DriftSeverity,
    BiometricAction,
    # Dataclasses
    BiometricTemplate,
    EnrollmentResult,
    VerificationResult,
    LivenessResult,
    DriftReport,
    EmbeddingResult,
    BiometricAuditEntry,
    BiometricConfig,
    # Constants
    EMBEDDING_VERSION,
    SPECTRAL_EMBEDDING_DIM,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_LIVENESS_THRESHOLD,
    DRIFT_SHORT_TERM_THRESHOLD,
    DRIFT_LONG_TERM_THRESHOLD,
)

# Embeddings
from .embeddings import (
    EmbeddingExtractor,
    SpectralFingerprint,
    NeuralEmbedding,
    get_extractor,
    serialize_embedding,
    deserialize_embedding,
    compute_embedding_hash,
)

# Liveness Detection
from .liveness import (
    LivenessDetector,
    LivenessCheck,
    ReplayDetector,
    DeepfakeDetector,
    BreathDetector,
    ChannelDetector,
    ProsodyDetector,
    CheckResult,
    check_liveness,
)

# Drift Monitoring
from .drift import (
    DriftMonitor,
    DriftSample,
    DriftTrend,
    AdaptiveTemplateUpdater,
    analyze_drift,
)

# Storage
from .storage import (
    BiometricStorage,
    get_biometric_storage,
    set_biometric_storage,
    BIOMETRIC_SCHEMA,
)

# Consent
from .consent import (
    BiometricConsentManager,
    ConsentType,
    ConsentStatus,
    ConsentToken,
    ConsentCheckResult,
    get_consent_manager,
    set_consent_manager,
)

# Main Service
from .service import (
    VoiceBiometricService,
    get_biometric_service,
    set_biometric_service,
    enroll_voice,
    verify_voice,
)

__all__ = [
    # Models - Enums
    "EmbeddingBackend",
    "VerificationStatus",
    "EnrollmentStatus",
    "LivenessStatus",
    "DriftSeverity",
    "BiometricAction",
    # Models - Dataclasses
    "BiometricTemplate",
    "EnrollmentResult",
    "VerificationResult",
    "LivenessResult",
    "DriftReport",
    "EmbeddingResult",
    "BiometricAuditEntry",
    "BiometricConfig",
    # Models - Constants
    "EMBEDDING_VERSION",
    "SPECTRAL_EMBEDDING_DIM",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_LIVENESS_THRESHOLD",
    "DRIFT_SHORT_TERM_THRESHOLD",
    "DRIFT_LONG_TERM_THRESHOLD",
    # Embeddings
    "EmbeddingExtractor",
    "SpectralFingerprint",
    "NeuralEmbedding",
    "get_extractor",
    "serialize_embedding",
    "deserialize_embedding",
    "compute_embedding_hash",
    # Liveness
    "LivenessDetector",
    "LivenessCheck",
    "ReplayDetector",
    "DeepfakeDetector",
    "BreathDetector",
    "ChannelDetector",
    "ProsodyDetector",
    "CheckResult",
    "check_liveness",
    # Drift
    "DriftMonitor",
    "DriftSample",
    "DriftTrend",
    "AdaptiveTemplateUpdater",
    "analyze_drift",
    # Storage
    "BiometricStorage",
    "get_biometric_storage",
    "set_biometric_storage",
    "BIOMETRIC_SCHEMA",
    # Consent
    "BiometricConsentManager",
    "ConsentType",
    "ConsentStatus",
    "ConsentToken",
    "ConsentCheckResult",
    "get_consent_manager",
    "set_consent_manager",
    # Service
    "VoiceBiometricService",
    "get_biometric_service",
    "set_biometric_service",
    "enroll_voice",
    "verify_voice",
]

__version__ = "0.10.0"
