"""
VØX SDK
-------

High-level Python SDK for the VØX voice synthesis platform.

Features:
    - Async-first design with sync wrappers
    - Automatic retry with exponential backoff
    - Session management with metrics
    - Workflow helpers for common operations
    - Comprehensive error handling

Quick Start:
    >>> from axiom_vox.sdk import VoxClient
    >>>
    >>> async with VoxClient(api_key="...") as vox:
    ...     audio = await vox.synthesize("Hello world", voice="warm")
    ...     print(f"Generated {len(audio)} bytes")

AXIØM Phase 8: Integrate - "How do the parts connect?"
"""

from .errors import (
    # Base
    VoxError,
    ErrorCategory,
    ErrorContext,
    RetryStrategy,
    # Auth
    AuthenticationError,
    AuthorizationError,
    # Validation
    ValidationError,
    InvalidVoiceError,
    InvalidTextError,
    InvalidAudioError,
    # Governance
    GovernanceError,
    ContentBlockedError,
    PolicyViolationError,
    RateLimitError,
    QuotaExceededError,
    # Biometric
    BiometricError,
    EnrollmentError,
    VerificationError,
    LivenessError,
    NotEnrolledError,
    # Synthesis
    SynthesisError,
    QualityGateError,
    SynthesisTimeoutError,
    # Resource
    ResourceError,
    ResourceUnavailableError,
    QueueFullError,
    # Network
    NetworkError,
    TimeoutError,
    ConnectionError,
    # Helpers
    from_http_response,
)

from .config import (
    VoxConfig,
    RetryConfig,
    TimeoutConfig,
    QualityConfig,
    GovernanceConfig,
    BiometricConfig,
    Environment,
    LogLevel,
    DEFAULT_CONFIGS,
    get_default_config,
)

from .retry import (
    RetryPolicy,
    RetryExecutor,
    RetryState,
    RateLimitHandler,
    with_retry,
    get_rate_limit_handler,
)

from .session import (
    VoxSession,
    SessionPool,
    SessionState,
    SessionMetrics,
    RequestContext,
)

from .workflows import (
    # Status
    WorkflowStatus,
    WorkflowStep,
    WorkflowResult,
    # Results
    SynthesisResult,
    EnrollmentResult,
    VerificationResult,
    DialogueLine,
    DialogueResult,
    # Synthesis workflows
    synthesize_with_quality_check,
    synthesize_verified,
    synthesize_batch,
    # Biometric workflows
    enroll_and_verify,
    continuous_verification,
    # Dialogue workflows
    synthesize_dialogue,
    # Builder
    WorkflowBuilder,
    Workflow,
)

from .client import (
    VoxClient,
)


__all__ = [
    # Main client
    "VoxClient",
    # Errors
    "VoxError",
    "ErrorCategory",
    "ErrorContext",
    "RetryStrategy",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "InvalidVoiceError",
    "InvalidTextError",
    "InvalidAudioError",
    "GovernanceError",
    "ContentBlockedError",
    "PolicyViolationError",
    "RateLimitError",
    "QuotaExceededError",
    "BiometricError",
    "EnrollmentError",
    "VerificationError",
    "LivenessError",
    "NotEnrolledError",
    "SynthesisError",
    "QualityGateError",
    "SynthesisTimeoutError",
    "ResourceError",
    "ResourceUnavailableError",
    "QueueFullError",
    "NetworkError",
    "TimeoutError",
    "ConnectionError",
    "from_http_response",
    # Config
    "VoxConfig",
    "RetryConfig",
    "TimeoutConfig",
    "QualityConfig",
    "GovernanceConfig",
    "BiometricConfig",
    "Environment",
    "LogLevel",
    "DEFAULT_CONFIGS",
    "get_default_config",
    # Retry
    "RetryPolicy",
    "RetryExecutor",
    "RetryState",
    "RateLimitHandler",
    "with_retry",
    "get_rate_limit_handler",
    # Session
    "VoxSession",
    "SessionPool",
    "SessionState",
    "SessionMetrics",
    "RequestContext",
    # Workflows
    "WorkflowStatus",
    "WorkflowStep",
    "WorkflowResult",
    "SynthesisResult",
    "EnrollmentResult",
    "VerificationResult",
    "DialogueLine",
    "DialogueResult",
    "synthesize_with_quality_check",
    "synthesize_verified",
    "synthesize_batch",
    "enroll_and_verify",
    "continuous_verification",
    "synthesize_dialogue",
    "WorkflowBuilder",
    "Workflow",
]
