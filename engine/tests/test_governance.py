"""
Tests for AXIØM VØX Resource Governance
---------------------------------------

Resource governance layer tests.

v0.12.0: Resource Governance Layer
"""

import time
import pytest
import threading
from unittest.mock import MagicMock, patch


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_user_id():
    """Sample user ID for testing."""
    return "test_user_123"


@pytest.fixture
def test_text():
    """Sample text for policy testing."""
    return "Hello, this is a test of the governance system."


# ============================================================================
# Rate Limiter Tests
# ============================================================================


class TestSlidingWindowRateLimiter:
    """Tests for sliding window rate limiter."""

    def test_allows_within_limit(self, test_user_id):
        """Test that requests within limit are allowed."""
        from axiom_vox.governance import SlidingWindowRateLimiter, RateLimitConfig

        config = RateLimitConfig(requests_per_minute=10)
        limiter = SlidingWindowRateLimiter(config=config)

        # First request should be allowed
        result = limiter.check(test_user_id)
        assert result.allowed is True
        assert result.remaining == 9

    def test_blocks_over_limit(self, test_user_id):
        """Test that requests over limit are blocked."""
        from axiom_vox.governance import SlidingWindowRateLimiter, RateLimitConfig

        config = RateLimitConfig(requests_per_minute=3)
        limiter = SlidingWindowRateLimiter(config=config)

        # Exhaust the limit
        for _ in range(3):
            limiter.check(test_user_id)

        # Next request should be blocked
        result = limiter.check(test_user_id)
        assert result.allowed is False
        assert result.retry_after is not None

    def test_reset_clears_bucket(self, test_user_id):
        """Test that reset clears the rate limit."""
        from axiom_vox.governance import SlidingWindowRateLimiter, RateLimitConfig

        config = RateLimitConfig(requests_per_minute=2)
        limiter = SlidingWindowRateLimiter(config=config)

        # Use up limit
        limiter.check(test_user_id)
        limiter.check(test_user_id)

        # Reset
        limiter.reset(test_user_id)

        # Should be allowed again
        result = limiter.check(test_user_id)
        assert result.allowed is True


class TestTokenBucketRateLimiter:
    """Tests for token bucket rate limiter."""

    def test_allows_burst(self, test_user_id):
        """Test that burst requests are allowed."""
        from axiom_vox.governance import TokenBucketRateLimiter, RateLimitConfig

        config = RateLimitConfig(burst_size=5, requests_per_minute=60)
        limiter = TokenBucketRateLimiter(config=config)

        # Burst of 5 should all be allowed
        for i in range(5):
            result = limiter.check(test_user_id)
            assert result.allowed is True

    def test_blocks_after_burst_exhausted(self, test_user_id):
        """Test blocking after burst is used."""
        from axiom_vox.governance import TokenBucketRateLimiter, RateLimitConfig

        config = RateLimitConfig(burst_size=2, requests_per_minute=60)
        limiter = TokenBucketRateLimiter(config=config)

        # Use burst
        limiter.check(test_user_id)
        limiter.check(test_user_id)

        # Should be blocked
        result = limiter.check(test_user_id)
        assert result.allowed is False

    def test_tokens_refill(self, test_user_id):
        """Test that tokens refill over time."""
        from axiom_vox.governance import TokenBucketRateLimiter, RateLimitConfig

        config = RateLimitConfig(burst_size=1, requests_per_minute=6000)  # 100/second
        limiter = TokenBucketRateLimiter(config=config)

        # Use the token
        limiter.check(test_user_id)

        # Should be blocked immediately
        result = limiter.check(test_user_id)
        assert result.allowed is False

        # Wait for refill (simulated by manipulating bucket)
        limiter._buckets[test_user_id].tokens = 1.0

        # Should be allowed again
        result = limiter.check(test_user_id)
        assert result.allowed is True


class TestTieredRateLimiter:
    """Tests for tiered rate limiter."""

    def test_free_tier_limits(self, test_user_id):
        """Test free tier has lower limits."""
        from axiom_vox.governance.rate_limiter import TieredRateLimiter

        limiter = TieredRateLimiter()

        # Default is free tier
        assert limiter.get_user_tier(test_user_id) == "free"

        result = limiter.check(test_user_id, operation="synthesize")
        assert result.allowed is True

    def test_tier_upgrade(self, test_user_id):
        """Test upgrading user tier."""
        from axiom_vox.governance.rate_limiter import TieredRateLimiter

        limiter = TieredRateLimiter()

        limiter.set_user_tier(test_user_id, "pro")
        assert limiter.get_user_tier(test_user_id) == "pro"

    def test_invalid_tier_raises(self, test_user_id):
        """Test that invalid tier raises error."""
        from axiom_vox.governance.rate_limiter import TieredRateLimiter

        limiter = TieredRateLimiter()

        with pytest.raises(ValueError):
            limiter.set_user_tier(test_user_id, "invalid_tier")


# ============================================================================
# Resource Manager Tests
# ============================================================================


class TestResourceManager:
    """Tests for resource manager."""

    def test_allocate_resources(self):
        """Test allocating resources."""
        from axiom_vox.governance import ResourceManager, ResourceType

        manager = ResourceManager()

        allocation = manager.allocate(
            request_id="req_123",
            resource_type=ResourceType.MEMORY,
            amount=512,
        )

        assert allocation.request_id == "req_123"
        assert allocation.amount == 512
        assert allocation.released is False

    def test_release_resources(self):
        """Test releasing resources."""
        from axiom_vox.governance import ResourceManager, ResourceType

        manager = ResourceManager()

        allocation = manager.allocate(
            request_id="req_123",
            resource_type=ResourceType.MEMORY,
            amount=512,
        )

        result = manager.release(allocation.allocation_id)
        assert result is True
        assert allocation.released is True

    def test_allocation_failure_insufficient_resources(self):
        """Test allocation fails when resources insufficient."""
        from axiom_vox.governance import ResourceManager, ResourceType, ResourceConfig
        from axiom_vox.governance.resource_manager import AllocationError

        config = ResourceConfig(max_memory_mb=100)
        manager = ResourceManager(config=config)

        with pytest.raises(AllocationError):
            manager.allocate(
                request_id="req_123",
                resource_type=ResourceType.MEMORY,
                amount=200,  # More than max
            )

    def test_get_status(self):
        """Test getting resource status."""
        from axiom_vox.governance import ResourceManager, ResourceType

        manager = ResourceManager()
        status = manager.get_status(ResourceType.GPU)

        assert status.resource_type == ResourceType.GPU
        assert status.used == 0
        assert status.available > 0


class TestQueueManager:
    """Tests for queue manager."""

    def test_enqueue_request(self):
        """Test enqueueing a request."""
        from axiom_vox.governance import QueueManager

        manager = QueueManager()

        request = manager.enqueue("req_123", priority=1)

        assert request.request_id == "req_123"
        assert request.priority == 1

    def test_dequeue_request(self):
        """Test dequeueing a request."""
        from axiom_vox.governance import QueueManager

        manager = QueueManager()
        manager.enqueue("req_123", priority=1)

        request = manager.dequeue(timeout=1.0)

        assert request is not None
        assert request.request_id == "req_123"

    def test_priority_ordering(self):
        """Test that higher priority is dequeued first."""
        from axiom_vox.governance import QueueManager

        manager = QueueManager()

        manager.enqueue("low_priority", priority=10)
        manager.enqueue("high_priority", priority=1)
        manager.enqueue("medium_priority", priority=5)

        # Should get highest priority first
        first = manager.dequeue(timeout=1.0)
        assert first.request_id == "high_priority"

    def test_queue_full_error(self):
        """Test queue full raises error."""
        from axiom_vox.governance import QueueManager, QueueConfig
        from axiom_vox.governance.resource_manager import QueueFullError

        config = QueueConfig(max_depth=2)
        manager = QueueManager(config=config)

        manager.enqueue("req_1")
        manager.enqueue("req_2")

        with pytest.raises(QueueFullError):
            manager.enqueue("req_3")


# ============================================================================
# Policy Engine Tests
# ============================================================================


class TestContentFilter:
    """Tests for content filter."""

    def test_allows_normal_content(self, test_text):
        """Test that normal content is allowed."""
        from axiom_vox.governance import PolicyEngine

        engine = PolicyEngine()
        result = engine.evaluate_content(test_text)

        assert result.allowed is True
        assert len(result.violations) == 0

    def test_blocks_pii(self):
        """Test that PII is detected."""
        from axiom_vox.governance import PolicyEngine

        engine = PolicyEngine()

        # Text with email
        text = "Contact me at test@example.com for more info"
        result = engine.evaluate_content(text)

        # Should warn about PII
        assert len(result.warnings) > 0 or "pii" in str(result.metadata).lower()

    def test_blocks_too_long_text(self):
        """Test that too-long text is blocked."""
        from axiom_vox.governance import PolicyEngine, ContentPolicy

        policy = ContentPolicy(max_text_length=50)
        engine = PolicyEngine(content_policy=policy)

        long_text = "x" * 100
        result = engine.evaluate_content(long_text)

        assert result.allowed is False
        assert any("long" in v.lower() for v in result.violations)

    def test_blocks_too_short_text(self):
        """Test that empty text is blocked."""
        from axiom_vox.governance import PolicyEngine

        engine = PolicyEngine()
        result = engine.evaluate_content("")

        assert result.allowed is False


class TestUsageValidator:
    """Tests for usage validator."""

    def test_allows_valid_operation(self, test_user_id):
        """Test that valid operations are allowed."""
        from axiom_vox.governance import PolicyEngine

        engine = PolicyEngine()
        result = engine.evaluate_usage(
            operation="synthesize",
            user_id=test_user_id,
            has_consent=True,
        )

        assert result.allowed is True

    def test_blocks_unauthorized_operation(self, test_user_id):
        """Test that unauthorized operations are blocked."""
        from axiom_vox.governance import PolicyEngine, UsagePolicy

        policy = UsagePolicy(allowed_operations={"synthesize"})
        engine = PolicyEngine(usage_policy=policy)

        result = engine.evaluate_usage(
            operation="delete",  # Not in allowed set
            user_id=test_user_id,
        )

        assert result.allowed is False

    def test_blocks_commercial_without_consent(self, test_user_id):
        """Test commercial use blocked without proper consent."""
        from axiom_vox.governance import PolicyEngine, UsagePolicy

        policy = UsagePolicy(allow_commercial=False)
        engine = PolicyEngine(usage_policy=policy)

        result = engine.evaluate_usage(
            operation="synthesize",
            user_id=test_user_id,
            is_commercial=True,
        )

        assert result.allowed is False


# ============================================================================
# Quota Manager Tests
# ============================================================================


class TestQuotaManager:
    """Tests for quota manager."""

    def test_check_quota(self, test_user_id):
        """Test checking quota status."""
        from axiom_vox.governance import QuotaManager

        manager = QuotaManager()
        status = manager.check_synthesis(test_user_id, text_length=100)

        assert "synthesis_day" in status
        assert status["synthesis_day"].exceeded is False

    def test_consume_quota(self, test_user_id):
        """Test consuming quota."""
        from axiom_vox.governance import QuotaManager

        manager = QuotaManager()

        initial = manager.check_synthesis(test_user_id, text_length=100)
        initial_remaining = initial["synthesis_day"].remaining

        manager.consume_synthesis(test_user_id, text_length=100, audio_duration_seconds=5.0)

        final = manager.check_synthesis(test_user_id, text_length=100)
        assert final["synthesis_day"].used == 1

    def test_quota_exceeded_error(self, test_user_id):
        """Test quota exceeded raises error."""
        from axiom_vox.governance import QuotaManager, QuotaConfig
        from axiom_vox.governance.quota_manager import QuotaExceededError

        config = QuotaConfig(synthesis_per_day=1)
        manager = QuotaManager(config=config)

        # Use up quota
        manager.consume_synthesis(test_user_id, text_length=100, audio_duration_seconds=1.0)

        # Should raise on next consume
        with pytest.raises(QuotaExceededError):
            manager.consume_synthesis(test_user_id, text_length=100, audio_duration_seconds=1.0)

    def test_tier_based_quotas(self, test_user_id):
        """Test tier-based quota limits."""
        from axiom_vox.governance import QuotaManager

        manager = QuotaManager()

        # Free tier
        assert manager.get_user_tier(test_user_id) == "free"

        # Upgrade to pro
        manager.set_user_tier(test_user_id, "pro")
        assert manager.get_user_tier(test_user_id) == "pro"


# ============================================================================
# Security Manager Tests
# ============================================================================


class TestAccessController:
    """Tests for access controller."""

    def test_register_and_validate_api_key(self, test_user_id):
        """Test API key registration and validation."""
        from axiom_vox.governance import AccessController

        controller = AccessController()
        api_key = "test_key_12345"

        controller.register_api_key(
            api_key=api_key,
            user_id=test_user_id,
            scopes={"synthesize", "verify"},
        )

        key_data = controller.validate_api_key(api_key)
        assert key_data["user_id"] == test_user_id
        assert "synthesize" in key_data["scopes"]

    def test_invalid_api_key_raises(self, test_user_id):
        """Test that invalid API key raises error."""
        from axiom_vox.governance import AccessController
        from axiom_vox.governance.security import AuthenticationError

        controller = AccessController()

        with pytest.raises(AuthenticationError):
            controller.validate_api_key("invalid_key")

    def test_create_and_validate_token(self, test_user_id):
        """Test token creation and validation."""
        from axiom_vox.governance import AccessController

        controller = AccessController()

        token = controller.create_token(
            user_id=test_user_id,
            scopes={"synthesize"},
            expires_in_seconds=3600,
        )

        validated = controller.validate_token(token.token_id)
        assert validated.user_id == test_user_id

    def test_revoke_token(self, test_user_id):
        """Test token revocation."""
        from axiom_vox.governance import AccessController
        from axiom_vox.governance.security import AuthenticationError

        controller = AccessController()

        token = controller.create_token(
            user_id=test_user_id,
            scopes={"synthesize"},
        )

        controller.revoke_token(token.token_id)

        with pytest.raises(AuthenticationError):
            controller.validate_token(token.token_id)

    def test_check_permission(self, test_user_id):
        """Test permission checking."""
        from axiom_vox.governance import AccessController

        controller = AccessController()

        # Default user role should have synthesize permission
        assert controller.check_permission(test_user_id, "synthesize") is True

        # But not admin permissions
        assert controller.check_permission(test_user_id, "manage_users") is False


class TestAuditLogger:
    """Tests for audit logger."""

    def test_log_entry(self, test_user_id):
        """Test logging audit entry."""
        from axiom_vox.governance import AuditLogger

        logger = AuditLogger()

        entry = logger.log(
            user_id=test_user_id,
            action="synthesize",
            resource_type="voice",
            resource_id="voice_123",
            result="success",
        )

        assert entry.user_id == test_user_id
        assert entry.action == "synthesize"
        assert entry.result == "success"

    def test_get_entries_with_filter(self, test_user_id):
        """Test retrieving entries with filter."""
        from axiom_vox.governance import AuditLogger

        logger = AuditLogger()

        # Log some entries
        logger.log(test_user_id, "synthesize", "voice", "v1", "success")
        logger.log(test_user_id, "verify", "biometric", "b1", "success")
        logger.log("other_user", "synthesize", "voice", "v2", "success")

        # Filter by user
        entries = logger.get_entries(user_id=test_user_id)
        assert len(entries) == 2
        assert all(e.user_id == test_user_id for e in entries)

        # Filter by action
        entries = logger.get_entries(action="verify")
        assert len(entries) == 1

    def test_callback_on_log(self, test_user_id):
        """Test callback is called on log."""
        from axiom_vox.governance import AuditLogger

        logger = AuditLogger()
        callback_entries = []

        logger.add_callback(lambda e: callback_entries.append(e))

        logger.log(test_user_id, "test", "test", "t1", "success")

        assert len(callback_entries) == 1
        assert callback_entries[0].action == "test"


# ============================================================================
# Integration Tests
# ============================================================================


class TestGovernanceIntegration:
    """Integration tests for governance module."""

    def test_imports_work(self):
        """Test all governance module imports."""
        from axiom_vox.governance import (
            # Enums
            PolicyType,
            PolicyAction,
            ResourceType,
            QuotaPeriod,
            SecurityLevel,
            # Rate Limiting
            SlidingWindowRateLimiter,
            TokenBucketRateLimiter,
            get_rate_limiter,
            # Resources
            ResourceManager,
            QueueManager,
            get_resource_manager,
            # Policies
            PolicyEngine,
            get_policy_engine,
            # Quotas
            QuotaManager,
            get_quota_manager,
            # Security
            SecurityManager,
            get_security_manager,
            # Defaults
            RATE_LIMIT_DEFAULTS,
            QUOTA_DEFAULTS,
        )

        assert SlidingWindowRateLimiter is not None
        assert PolicyEngine is not None

    def test_main_module_exports(self):
        """Test exports from main axiom_vox module."""
        from axiom_vox import (
            PolicyType,
            RateLimitConfig,
            SlidingWindowRateLimiter,
            ResourceManager,
            PolicyEngine,
            QuotaManager,
            SecurityManager,
            RATE_LIMIT_DEFAULTS,
            QUOTA_DEFAULTS,
        )

        assert PolicyType is not None
        assert RATE_LIMIT_DEFAULTS is not None

    def test_default_configs(self):
        """Test default configurations are valid."""
        from axiom_vox.governance import (
            RATE_LIMIT_DEFAULTS,
            QUOTA_DEFAULTS,
            RESOURCE_LIMITS,
        )

        # All tiers should be present
        assert "free" in RATE_LIMIT_DEFAULTS
        assert "basic" in RATE_LIMIT_DEFAULTS
        assert "pro" in RATE_LIMIT_DEFAULTS
        assert "enterprise" in RATE_LIMIT_DEFAULTS

        # Quotas should have same tiers
        assert set(QUOTA_DEFAULTS.keys()) == set(RATE_LIMIT_DEFAULTS.keys())

        # Resource limits should be defined
        assert "synthesis" in RESOURCE_LIMITS

    def test_complete_governance_flow(self, test_user_id, test_text):
        """Test complete governance flow."""
        from axiom_vox.governance import (
            SlidingWindowRateLimiter,
            PolicyEngine,
            QuotaManager,
            SecurityManager,
        )

        # 1. Check rate limit
        limiter = SlidingWindowRateLimiter()
        rate_result = limiter.check(test_user_id)
        assert rate_result.allowed is True

        # 2. Check policies
        policy = PolicyEngine()
        policy_results = policy.evaluate_all(
            text=test_text,
            operation="synthesize",
            user_id=test_user_id,
            has_consent=True,
        )
        assert all(r.allowed for r in policy_results)

        # 3. Check quotas
        quotas = QuotaManager()
        allowed, exceeded = quotas.is_allowed(
            user_id=test_user_id,
            operation="synthesize",
            text_length=len(test_text),
        )
        assert allowed is True

        # 4. Log audit
        security = SecurityManager()
        security.audit.log_synthesis(
            user_id=test_user_id,
            voice_id="test_voice",
            text_length=len(test_text),
            result="success",
        )


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_rate_limiter_thread_safe(self, test_user_id):
        """Test rate limiter under concurrent access."""
        from axiom_vox.governance import SlidingWindowRateLimiter, RateLimitConfig

        config = RateLimitConfig(requests_per_minute=100)
        limiter = SlidingWindowRateLimiter(config=config)

        results = []
        errors = []

        def check_rate_limit():
            try:
                for _ in range(10):
                    result = limiter.check(test_user_id)
                    results.append(result.allowed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check_rate_limit) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
