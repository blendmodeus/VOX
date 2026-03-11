"""
VØX SDK Errors
--------------

Unified exception hierarchy for VØX SDK.

All SDK errors inherit from VoxError, providing:
    - Consistent error handling across all operations
    - Rich context for debugging
    - Governance information when applicable
    - Retry guidance for transient errors

AXIØM Phase 8: Integrate - "How do the parts connect?"
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class ErrorCategory(str, Enum):
    """Categories of VØX errors."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    GOVERNANCE = "governance"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    RESOURCE = "resource"
    BIOMETRIC = "biometric"
    SYNTHESIS = "synthesis"
    NETWORK = "network"
    INTERNAL = "internal"


class RetryStrategy(str, Enum):
    """Retry strategies for errors."""
    NO_RETRY = "no_retry"
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    WAIT_AND_RETRY = "wait_and_retry"


@dataclass
class ErrorContext:
    """Rich context for error debugging."""
    request_id: Optional[str] = None
    operation: Optional[str] = None
    voice_id: Optional[str] = None
    user_id: Optional[str] = None
    text_preview: Optional[str] = None
    governance_report: Optional[Dict[str, Any]] = None
    rate_limit_info: Optional[Dict[str, Any]] = None
    quota_info: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)


class VoxError(Exception):
    """
    Base exception for all VØX SDK errors.

    Provides:
        - Error categorization
        - Rich context
        - Retry guidance
        - Human-readable messages
    """

    category: ErrorCategory = ErrorCategory.INTERNAL
    retry_strategy: RetryStrategy = RetryStrategy.NO_RETRY
    http_status: int = 500

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        retry_after: Optional[float] = None,
    ):
        """
        Initialize VoxError.

        Args:
            message: Human-readable error message
            context: Rich error context
            cause: Original exception that caused this error
            retry_after: Seconds to wait before retry (if applicable)
        """
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext()
        self.cause = cause
        self.retry_after = retry_after

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "category": self.category.value,
            "message": self.message,
            "retry_strategy": self.retry_strategy.value,
            "retry_after": self.retry_after,
            "context": {
                "request_id": self.context.request_id,
                "operation": self.context.operation,
                "voice_id": self.context.voice_id,
            },
        }

    @property
    def is_retryable(self) -> bool:
        """Check if this error is retryable."""
        return self.retry_strategy != RetryStrategy.NO_RETRY


# ============================================================================
# Authentication Errors
# ============================================================================


class AuthenticationError(VoxError):
    """Authentication failed (invalid credentials)."""

    category = ErrorCategory.AUTHENTICATION
    retry_strategy = RetryStrategy.NO_RETRY
    http_status = 401


class InvalidAPIKeyError(AuthenticationError):
    """Invalid or missing API key."""

    def __init__(self, message: str = "Invalid or missing API key", **kwargs):
        super().__init__(message, **kwargs)


class ExpiredTokenError(AuthenticationError):
    """Access token has expired."""

    retry_strategy = RetryStrategy.IMMEDIATE  # Get new token

    def __init__(self, message: str = "Access token has expired", **kwargs):
        super().__init__(message, **kwargs)


# ============================================================================
# Authorization Errors
# ============================================================================


class AuthorizationError(VoxError):
    """Authorization failed (insufficient permissions)."""

    category = ErrorCategory.AUTHORIZATION
    retry_strategy = RetryStrategy.NO_RETRY
    http_status = 403


class InsufficientScopeError(AuthorizationError):
    """Token lacks required scope."""

    def __init__(
        self,
        required_scope: str,
        available_scopes: List[str],
        **kwargs,
    ):
        message = f"Scope '{required_scope}' required. Available: {available_scopes}"
        super().__init__(message, **kwargs)
        self.required_scope = required_scope
        self.available_scopes = available_scopes


class ConsentRequiredError(AuthorizationError):
    """Operation requires consent that hasn't been granted."""

    def __init__(
        self,
        voice_id: str,
        required_scopes: List[str],
        **kwargs,
    ):
        message = f"Consent required for voice '{voice_id}': {required_scopes}"
        super().__init__(message, **kwargs)
        self.voice_id = voice_id
        self.required_scopes = required_scopes


# ============================================================================
# Validation Errors
# ============================================================================


class ValidationError(VoxError):
    """Request validation failed."""

    category = ErrorCategory.VALIDATION
    retry_strategy = RetryStrategy.NO_RETRY
    http_status = 400


class InvalidTextError(ValidationError):
    """Text input is invalid."""

    def __init__(
        self,
        reason: str,
        text_length: Optional[int] = None,
        **kwargs,
    ):
        message = f"Invalid text: {reason}"
        super().__init__(message, **kwargs)
        self.reason = reason
        self.text_length = text_length


class InvalidVoiceError(ValidationError):
    """Voice ID is invalid or not found."""

    def __init__(self, voice_id: str, **kwargs):
        message = f"Voice not found: '{voice_id}'"
        super().__init__(message, **kwargs)
        self.voice_id = voice_id


class InvalidAudioError(ValidationError):
    """Audio input is invalid."""

    def __init__(self, reason: str, **kwargs):
        message = f"Invalid audio: {reason}"
        super().__init__(message, **kwargs)
        self.reason = reason


# ============================================================================
# Governance Errors
# ============================================================================


class GovernanceError(VoxError):
    """Governance check failed."""

    category = ErrorCategory.GOVERNANCE
    retry_strategy = RetryStrategy.NO_RETRY
    http_status = 422


class ContentBlockedError(GovernanceError):
    """Content was blocked by governance."""

    def __init__(
        self,
        reason: str,
        violations: Optional[List[str]] = None,
        governance_report: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        message = f"Content blocked: {reason}"
        context = kwargs.pop("context", None) or ErrorContext()
        context.governance_report = governance_report
        super().__init__(message, context=context, **kwargs)
        self.reason = reason
        self.violations = violations or []


class PolicyViolationError(GovernanceError):
    """Request violates policy."""

    def __init__(
        self,
        policy_id: str,
        violations: List[str],
        **kwargs,
    ):
        message = f"Policy violation ({policy_id}): {', '.join(violations)}"
        super().__init__(message, **kwargs)
        self.policy_id = policy_id
        self.violations = violations


# ============================================================================
# Rate Limit Errors
# ============================================================================


class RateLimitError(VoxError):
    """Rate limit exceeded."""

    category = ErrorCategory.RATE_LIMIT
    retry_strategy = RetryStrategy.WAIT_AND_RETRY
    http_status = 429

    def __init__(
        self,
        limit: int,
        remaining: int,
        reset_at: float,
        retry_after: float,
        **kwargs,
    ):
        message = f"Rate limit exceeded ({remaining}/{limit}). Retry after {retry_after:.1f}s"
        context = kwargs.pop("context", None) or ErrorContext()
        context.rate_limit_info = {
            "limit": limit,
            "remaining": remaining,
            "reset_at": reset_at,
        }
        super().__init__(message, context=context, retry_after=retry_after, **kwargs)
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at


# ============================================================================
# Quota Errors
# ============================================================================


class QuotaExceededError(VoxError):
    """Usage quota exceeded."""

    category = ErrorCategory.QUOTA
    retry_strategy = RetryStrategy.WAIT_AND_RETRY
    http_status = 429

    def __init__(
        self,
        quota_name: str,
        limit: int,
        used: int,
        resets_at: float,
        **kwargs,
    ):
        import time
        retry_after = max(0, resets_at - time.time())
        message = f"Quota exceeded: {quota_name} ({used}/{limit}). Resets in {retry_after:.0f}s"
        context = kwargs.pop("context", None) or ErrorContext()
        context.quota_info = {
            "quota_name": quota_name,
            "limit": limit,
            "used": used,
            "resets_at": resets_at,
        }
        super().__init__(message, context=context, retry_after=retry_after, **kwargs)
        self.quota_name = quota_name
        self.limit = limit
        self.used = used
        self.resets_at = resets_at


# ============================================================================
# Resource Errors
# ============================================================================


class ResourceError(VoxError):
    """Resource allocation failed."""

    category = ErrorCategory.RESOURCE
    retry_strategy = RetryStrategy.EXPONENTIAL_BACKOFF
    http_status = 503


class ResourceUnavailableError(ResourceError):
    """Required resource is not available."""

    def __init__(
        self,
        resource_type: str,
        requested: float,
        available: float,
        **kwargs,
    ):
        message = f"Resource unavailable: {resource_type} (requested {requested}, available {available})"
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.requested = requested
        self.available = available


class QueueFullError(ResourceError):
    """Request queue is full."""

    def __init__(self, queue_depth: int, **kwargs):
        message = f"Queue full (depth: {queue_depth}). Try again later."
        super().__init__(message, **kwargs)
        self.queue_depth = queue_depth


# ============================================================================
# Biometric Errors
# ============================================================================


class BiometricError(VoxError):
    """Biometric operation failed."""

    category = ErrorCategory.BIOMETRIC
    retry_strategy = RetryStrategy.NO_RETRY
    http_status = 400


class EnrollmentError(BiometricError):
    """Voice enrollment failed."""

    def __init__(self, voice_id: str, reason: str, **kwargs):
        message = f"Enrollment failed for '{voice_id}': {reason}"
        super().__init__(message, **kwargs)
        self.voice_id = voice_id
        self.reason = reason


class VerificationError(BiometricError):
    """Voice verification failed."""

    def __init__(
        self,
        voice_id: str,
        similarity: float,
        threshold: float,
        **kwargs,
    ):
        message = f"Verification failed for '{voice_id}': similarity {similarity:.3f} < threshold {threshold:.3f}"
        super().__init__(message, **kwargs)
        self.voice_id = voice_id
        self.similarity = similarity
        self.threshold = threshold


class LivenessError(BiometricError):
    """Liveness check failed."""

    def __init__(self, reason: str, score: float, **kwargs):
        message = f"Liveness check failed: {reason} (score: {score:.3f})"
        super().__init__(message, **kwargs)
        self.reason = reason
        self.score = score


class NotEnrolledError(BiometricError):
    """Voice is not enrolled for biometric verification."""

    http_status = 404

    def __init__(self, voice_id: str, **kwargs):
        message = f"Voice '{voice_id}' is not enrolled for biometric verification"
        super().__init__(message, **kwargs)
        self.voice_id = voice_id


# ============================================================================
# Synthesis Errors
# ============================================================================


class SynthesisError(VoxError):
    """Synthesis operation failed."""

    category = ErrorCategory.SYNTHESIS
    retry_strategy = RetryStrategy.EXPONENTIAL_BACKOFF
    http_status = 500


class QualityGateError(SynthesisError):
    """Synthesis quality gate failed."""

    retry_strategy = RetryStrategy.IMMEDIATE

    def __init__(
        self,
        quality_score: float,
        min_required: float,
        **kwargs,
    ):
        message = f"Quality gate failed: {quality_score:.3f} < {min_required:.3f}"
        super().__init__(message, **kwargs)
        self.quality_score = quality_score
        self.min_required = min_required


class SynthesisTimeoutError(SynthesisError):
    """Synthesis timed out."""

    def __init__(self, timeout_seconds: float, **kwargs):
        message = f"Synthesis timed out after {timeout_seconds:.1f}s"
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds


# ============================================================================
# Network Errors
# ============================================================================


class NetworkError(VoxError):
    """Network communication failed."""

    category = ErrorCategory.NETWORK
    retry_strategy = RetryStrategy.EXPONENTIAL_BACKOFF
    http_status = 502


class ConnectionError(NetworkError):
    """Failed to connect to VØX service."""

    def __init__(self, url: str, reason: str, **kwargs):
        message = f"Connection failed to {url}: {reason}"
        super().__init__(message, **kwargs)
        self.url = url
        self.reason = reason


class TimeoutError(NetworkError):
    """Request timed out."""

    def __init__(self, timeout_seconds: float, **kwargs):
        message = f"Request timed out after {timeout_seconds:.1f}s"
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds


# ============================================================================
# Helper Functions
# ============================================================================


def from_http_response(
    status_code: int,
    response_body: Dict[str, Any],
    context: Optional[ErrorContext] = None,
) -> VoxError:
    """
    Create appropriate VoxError from HTTP response.

    Args:
        status_code: HTTP status code
        response_body: Response JSON body
        context: Error context

    Returns:
        Appropriate VoxError subclass
    """
    error_type = response_body.get("error", "")
    message = response_body.get("message", "Unknown error")
    details = response_body.get("details", {})

    # Map status codes to error classes
    if status_code == 401:
        return AuthenticationError(message, context=context)
    elif status_code == 403:
        return AuthorizationError(message, context=context)
    elif status_code == 400:
        return ValidationError(message, context=context)
    elif status_code == 404:
        return InvalidVoiceError(details.get("voice_id", "unknown"), context=context)
    elif status_code == 422:
        return GovernanceError(message, context=context)
    elif status_code == 429:
        if "quota" in error_type.lower():
            return QuotaExceededError(
                quota_name=details.get("quota_name", "unknown"),
                limit=details.get("limit", 0),
                used=details.get("used", 0),
                resets_at=details.get("resets_at", 0),
                context=context,
            )
        else:
            return RateLimitError(
                limit=details.get("limit", 0),
                remaining=details.get("remaining", 0),
                reset_at=details.get("reset_at", 0),
                retry_after=details.get("retry_after", 60),
                context=context,
            )
    elif status_code == 503:
        return ResourceUnavailableError(
            resource_type=details.get("resource_type", "unknown"),
            requested=details.get("requested", 0),
            available=details.get("available", 0),
            context=context,
        )
    else:
        return VoxError(message, context=context)
