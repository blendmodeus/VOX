"""
AXIØM VØX Governance Module
---------------------------

Resource governance and policy enforcement for voice operations.

AXIØM Phase 7: Constrain - "What limits clarify the solution?"

Components:
    - RateLimiter: Sliding window rate limiting
    - ResourceManager: GPU/memory/queue management
    - PolicyEngine: Content and usage policy enforcement
    - QuotaManager: Usage tracking and quotas
    - SecurityManager: Encryption and access control

v0.12.0: Resource Governance Layer
"""

from .models import (
    # Enums
    PolicyType,
    PolicyAction,
    ResourceType,
    QuotaPeriod,
    SecurityLevel,
    ViolationType,
    # Rate Limiting
    RateLimitConfig,
    RateLimitResult,
    RateLimitBucket,
    # Resources
    ResourceConfig,
    ResourceStatus,
    ResourceAllocation,
    QueueConfig,
    QueueStatus,
    # Policies
    Policy,
    PolicyResult,
    ContentPolicy,
    UsagePolicy,
    # Quotas
    QuotaConfig,
    QuotaStatus,
    QuotaUsage,
    # Security
    SecurityConfig,
    AccessToken,
    AuditEntry,
    # Thresholds
    RATE_LIMIT_DEFAULTS,
    RESOURCE_LIMITS,
    QUOTA_DEFAULTS,
)

from .rate_limiter import (
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    get_rate_limiter,
    set_rate_limiter,
)

from .resource_manager import (
    ResourceManager,
    QueueManager,
    get_resource_manager,
    set_resource_manager,
)

from .policy_engine import (
    PolicyEngine,
    ContentFilter,
    UsageValidator,
    get_policy_engine,
    set_policy_engine,
)

from .quota_manager import (
    QuotaManager,
    QuotaTracker,
    get_quota_manager,
    set_quota_manager,
)

from .security import (
    SecurityManager,
    AccessController,
    AuditLogger,
    get_security_manager,
    set_security_manager,
)

__all__ = [
    # Enums
    "PolicyType",
    "PolicyAction",
    "ResourceType",
    "QuotaPeriod",
    "SecurityLevel",
    "ViolationType",
    # Rate Limiting Models
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimitBucket",
    # Resource Models
    "ResourceConfig",
    "ResourceStatus",
    "ResourceAllocation",
    "QueueConfig",
    "QueueStatus",
    # Policy Models
    "Policy",
    "PolicyResult",
    "ContentPolicy",
    "UsagePolicy",
    # Quota Models
    "QuotaConfig",
    "QuotaStatus",
    "QuotaUsage",
    # Security Models
    "SecurityConfig",
    "AccessToken",
    "AuditEntry",
    # Thresholds
    "RATE_LIMIT_DEFAULTS",
    "RESOURCE_LIMITS",
    "QUOTA_DEFAULTS",
    # Rate Limiter
    "SlidingWindowRateLimiter",
    "TokenBucketRateLimiter",
    "get_rate_limiter",
    "set_rate_limiter",
    # Resource Manager
    "ResourceManager",
    "QueueManager",
    "get_resource_manager",
    "set_resource_manager",
    # Policy Engine
    "PolicyEngine",
    "ContentFilter",
    "UsageValidator",
    "get_policy_engine",
    "set_policy_engine",
    # Quota Manager
    "QuotaManager",
    "QuotaTracker",
    "get_quota_manager",
    "set_quota_manager",
    # Security
    "SecurityManager",
    "AccessController",
    "AuditLogger",
    "get_security_manager",
    "set_security_manager",
]
