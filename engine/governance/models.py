"""
Governance Models
-----------------

Data models for resource governance and policy enforcement.

AXIØM Phase 7: Constrain - "What limits clarify the solution?"
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import time


# ============================================================================
# Enums
# ============================================================================


class PolicyType(str, Enum):
    """Types of governance policies."""
    CONTENT = "content"          # Content-based restrictions
    USAGE = "usage"              # Usage pattern restrictions
    RATE = "rate"                # Rate limiting
    RESOURCE = "resource"        # Resource allocation
    SECURITY = "security"        # Security policies
    QUALITY = "quality"          # Quality requirements


class PolicyAction(str, Enum):
    """Actions to take on policy violation."""
    ALLOW = "allow"              # Allow the request
    WARN = "warn"                # Allow but log warning
    THROTTLE = "throttle"        # Slow down requests
    BLOCK = "block"              # Block the request
    QUEUE = "queue"              # Queue for later processing
    ESCALATE = "escalate"        # Escalate for review


class ResourceType(str, Enum):
    """Types of resources to manage."""
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    QUEUE = "queue"


class QuotaPeriod(str, Enum):
    """Quota measurement periods."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class SecurityLevel(str, Enum):
    """Security classification levels."""
    PUBLIC = "public"            # No restrictions
    INTERNAL = "internal"        # Internal use only
    CONFIDENTIAL = "confidential"  # Limited access
    RESTRICTED = "restricted"    # Strict access control


class ViolationType(str, Enum):
    """Types of policy violations."""
    RATE_EXCEEDED = "rate_exceeded"
    QUOTA_EXCEEDED = "quota_exceeded"
    CONTENT_BLOCKED = "content_blocked"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    ACCESS_DENIED = "access_denied"
    QUALITY_FAILED = "quality_failed"
    CONSENT_MISSING = "consent_missing"


# ============================================================================
# Rate Limiting Models
# ============================================================================


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    window_size_seconds: int = 60
    sliding_window: bool = True
    per_voice: bool = True
    per_user: bool = True
    per_ip: bool = False


@dataclass
class RateLimitBucket:
    """A rate limit bucket for tracking requests."""
    key: str
    window_start: float
    request_count: int = 0
    tokens: float = 0.0
    last_update: float = field(default_factory=time.time)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None
    limit: int = 0
    bucket_key: str = ""
    message: str = ""


# ============================================================================
# Resource Models
# ============================================================================


@dataclass
class ResourceConfig:
    """Configuration for resource management."""
    # GPU limits
    max_gpu_memory_mb: int = 4096
    max_gpu_utilization_percent: int = 80
    gpu_reservation_timeout_seconds: int = 30

    # CPU limits
    max_cpu_percent: int = 80
    max_threads: int = 4

    # Memory limits
    max_memory_mb: int = 2048
    memory_warning_threshold: float = 0.8

    # Queue limits
    max_queue_depth: int = 100
    max_queue_wait_seconds: int = 60
    priority_levels: int = 3

    # Timeouts
    synthesis_timeout_seconds: int = 30
    enrollment_timeout_seconds: int = 60
    verification_timeout_seconds: int = 10


@dataclass
class ResourceStatus:
    """Current status of a resource."""
    resource_type: ResourceType
    total: float
    used: float
    available: float
    utilization_percent: float
    healthy: bool = True
    message: str = ""


@dataclass
class ResourceAllocation:
    """An allocation of resources for a request."""
    allocation_id: str
    request_id: str
    resource_type: ResourceType
    amount: float
    allocated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    released: bool = False


@dataclass
class QueueConfig:
    """Configuration for request queuing."""
    max_depth: int = 100
    max_wait_seconds: int = 60
    priority_boost_per_second: float = 0.1
    fairness_enabled: bool = True
    preemption_enabled: bool = False


@dataclass
class QueueStatus:
    """Status of the request queue."""
    depth: int
    oldest_request_age_seconds: float
    average_wait_seconds: float
    processing_rate_per_second: float
    blocked: bool = False


# ============================================================================
# Policy Models
# ============================================================================


@dataclass
class Policy:
    """A governance policy."""
    policy_id: str
    name: str
    policy_type: PolicyType
    action: PolicyAction
    enabled: bool = True
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyResult:
    """Result of policy evaluation."""
    policy_id: str
    policy_type: PolicyType
    action: PolicyAction
    allowed: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentPolicy:
    """Content-specific policy rules."""
    blocked_patterns: List[str] = field(default_factory=list)
    blocked_topics: List[str] = field(default_factory=list)
    max_text_length: int = 10000
    min_text_length: int = 1
    require_language_detection: bool = False
    allowed_languages: List[str] = field(default_factory=lambda: ["en"])
    profanity_filter: bool = True
    pii_detection: bool = True


@dataclass
class UsagePolicy:
    """Usage-specific policy rules."""
    allowed_operations: Set[str] = field(default_factory=lambda: {
        "synthesize", "verify", "enroll", "stream"
    })
    require_consent: bool = True
    require_authentication: bool = True
    allow_commercial: bool = False
    allow_third_party: bool = False
    max_audio_duration_seconds: int = 300
    max_concurrent_requests: int = 5


# ============================================================================
# Quota Models
# ============================================================================


@dataclass
class QuotaConfig:
    """Configuration for usage quotas."""
    # Synthesis quotas
    synthesis_per_minute: int = 10
    synthesis_per_hour: int = 100
    synthesis_per_day: int = 1000

    # Character quotas
    characters_per_minute: int = 5000
    characters_per_hour: int = 50000
    characters_per_day: int = 500000

    # Audio quotas
    audio_seconds_per_hour: int = 3600
    audio_seconds_per_day: int = 36000

    # Storage quotas
    voice_slots: int = 10
    adapter_storage_mb: int = 1024

    # Biometric quotas
    enrollments_per_day: int = 5
    verifications_per_hour: int = 100


@dataclass
class QuotaStatus:
    """Status of quota usage."""
    quota_name: str
    period: QuotaPeriod
    limit: int
    used: int
    remaining: int
    resets_at: float
    exceeded: bool = False
    warning_threshold_reached: bool = False


@dataclass
class QuotaUsage:
    """Tracked quota usage."""
    user_id: str
    voice_id: Optional[str]
    quota_name: str
    period: QuotaPeriod
    period_start: float
    count: int = 0
    last_updated: float = field(default_factory=time.time)


# ============================================================================
# Security Models
# ============================================================================


@dataclass
class SecurityConfig:
    """Configuration for security management."""
    # Encryption
    encrypt_at_rest: bool = True
    encrypt_in_transit: bool = True
    encryption_algorithm: str = "AES-256-GCM"

    # Authentication
    require_api_key: bool = True
    api_key_rotation_days: int = 90
    session_timeout_seconds: int = 3600

    # Access control
    rbac_enabled: bool = True
    default_security_level: SecurityLevel = SecurityLevel.INTERNAL

    # Audit
    audit_enabled: bool = True
    audit_retention_days: int = 365
    audit_sensitive_operations: bool = True


@dataclass
class AccessToken:
    """An access token for authentication."""
    token_id: str
    user_id: str
    scopes: Set[str]
    issued_at: float
    expires_at: float
    revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEntry:
    """An entry in the audit log."""
    entry_id: str
    timestamp: float
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    result: str
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Default Thresholds
# ============================================================================


RATE_LIMIT_DEFAULTS = {
    "free": RateLimitConfig(
        requests_per_minute=10,
        requests_per_hour=100,
        requests_per_day=500,
        burst_size=3,
    ),
    "basic": RateLimitConfig(
        requests_per_minute=30,
        requests_per_hour=500,
        requests_per_day=5000,
        burst_size=5,
    ),
    "pro": RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=2000,
        requests_per_day=20000,
        burst_size=10,
    ),
    "enterprise": RateLimitConfig(
        requests_per_minute=300,
        requests_per_hour=10000,
        requests_per_day=100000,
        burst_size=50,
    ),
}


RESOURCE_LIMITS = {
    "synthesis": ResourceConfig(
        max_gpu_memory_mb=2048,
        max_memory_mb=1024,
        synthesis_timeout_seconds=30,
    ),
    "enrollment": ResourceConfig(
        max_gpu_memory_mb=4096,
        max_memory_mb=2048,
        enrollment_timeout_seconds=60,
    ),
    "streaming": ResourceConfig(
        max_gpu_memory_mb=1024,
        max_memory_mb=512,
        synthesis_timeout_seconds=60,
        max_queue_depth=50,
    ),
}


QUOTA_DEFAULTS = {
    "free": QuotaConfig(
        synthesis_per_minute=5,
        synthesis_per_hour=50,
        synthesis_per_day=200,
        characters_per_day=50000,
        audio_seconds_per_day=600,
        voice_slots=2,
    ),
    "basic": QuotaConfig(
        synthesis_per_minute=20,
        synthesis_per_hour=200,
        synthesis_per_day=2000,
        characters_per_day=500000,
        audio_seconds_per_day=7200,
        voice_slots=5,
    ),
    "pro": QuotaConfig(
        synthesis_per_minute=50,
        synthesis_per_hour=1000,
        synthesis_per_day=10000,
        characters_per_day=2000000,
        audio_seconds_per_day=36000,
        voice_slots=20,
    ),
    "enterprise": QuotaConfig(
        synthesis_per_minute=200,
        synthesis_per_hour=5000,
        synthesis_per_day=100000,
        characters_per_day=10000000,
        audio_seconds_per_day=360000,
        voice_slots=100,
    ),
}
