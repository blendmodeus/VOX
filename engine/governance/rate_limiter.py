"""
Rate Limiter
------------

Sliding window and token bucket rate limiting implementations.

AXIØM Phase 7: Constrain - "What limits clarify the solution?"
"""

import logging
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

from .models import (
    RateLimitConfig,
    RateLimitResult,
    RateLimitBucket,
    RATE_LIMIT_DEFAULTS,
)

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter with sub-window precision.

    Provides smooth rate limiting by tracking requests in
    multiple sub-windows and calculating a weighted count.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        sub_windows: int = 10,
    ):
        """
        Initialize sliding window rate limiter.

        Args:
            config: Rate limit configuration
            sub_windows: Number of sub-windows for precision
        """
        self.config = config or RateLimitConfig()
        self.sub_windows = sub_windows
        self._buckets: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = threading.RLock()

    def check(
        self,
        key: str,
        cost: int = 1,
        window_seconds: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> RateLimitResult:
        """
        Check if request is allowed and consume quota.

        Args:
            key: Rate limit key (e.g., "user:123" or "voice:abc")
            cost: Cost of this request (default 1)
            window_seconds: Window size (default from config)
            limit: Request limit (default from config)

        Returns:
            RateLimitResult
        """
        window_seconds = window_seconds or self.config.window_size_seconds
        limit = limit or self.config.requests_per_minute

        with self._lock:
            now = time.time()
            sub_window_size = window_seconds / self.sub_windows
            current_sub_window = int(now / sub_window_size)

            # Get bucket for this key
            bucket = self._buckets[key]

            # Clean old sub-windows
            min_valid_window = current_sub_window - self.sub_windows
            old_keys = [k for k in bucket.keys() if k < min_valid_window]
            for k in old_keys:
                del bucket[k]

            # Calculate weighted count
            total_count = 0
            window_start = now - window_seconds

            for sub_win, count in bucket.items():
                sub_win_start = sub_win * sub_window_size
                sub_win_end = sub_win_start + sub_window_size

                # Calculate overlap with current window
                overlap_start = max(sub_win_start, window_start)
                overlap_end = min(sub_win_end, now)

                if overlap_end > overlap_start:
                    weight = (overlap_end - overlap_start) / sub_window_size
                    total_count += count * weight

            # Check if allowed
            remaining = max(0, limit - int(total_count))
            allowed = total_count + cost <= limit

            if allowed:
                # Record this request
                bucket[current_sub_window] += cost
                remaining = max(0, remaining - cost)

            # Calculate reset time
            oldest_window = min(bucket.keys()) if bucket else current_sub_window
            reset_at = (oldest_window + self.sub_windows + 1) * sub_window_size

            retry_after = None
            if not allowed:
                retry_after = reset_at - now

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_at=reset_at,
                retry_after=retry_after,
                limit=limit,
                bucket_key=key,
                message="" if allowed else f"Rate limit exceeded. Retry after {retry_after:.1f}s",
            )

    def get_status(self, key: str) -> Dict[str, int]:
        """Get current rate limit status for a key."""
        with self._lock:
            bucket = self._buckets.get(key, {})
            return {
                "current_count": sum(bucket.values()),
                "sub_windows": len(bucket),
            }

    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limit for a key or all keys."""
        with self._lock:
            if key:
                self._buckets.pop(key, None)
            else:
                self._buckets.clear()


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for burst handling.

    Tokens refill at a constant rate, allowing bursts
    up to the bucket capacity.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize token bucket rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, RateLimitBucket] = {}
        self._lock = threading.RLock()

    def check(
        self,
        key: str,
        cost: int = 1,
        capacity: Optional[int] = None,
        refill_rate: Optional[float] = None,
    ) -> RateLimitResult:
        """
        Check if request is allowed and consume tokens.

        Args:
            key: Rate limit key
            cost: Token cost of this request
            capacity: Bucket capacity (default: burst_size)
            refill_rate: Tokens per second (default: requests_per_minute / 60)

        Returns:
            RateLimitResult
        """
        capacity = capacity or self.config.burst_size
        refill_rate = refill_rate or (self.config.requests_per_minute / 60.0)

        with self._lock:
            now = time.time()

            # Get or create bucket
            if key not in self._buckets:
                self._buckets[key] = RateLimitBucket(
                    key=key,
                    window_start=now,
                    tokens=float(capacity),
                    last_update=now,
                )

            bucket = self._buckets[key]

            # Refill tokens based on elapsed time
            elapsed = now - bucket.last_update
            bucket.tokens = min(capacity, bucket.tokens + elapsed * refill_rate)
            bucket.last_update = now

            # Check if we have enough tokens
            allowed = bucket.tokens >= cost

            if allowed:
                bucket.tokens -= cost

            remaining = int(bucket.tokens)

            # Calculate when we'll have enough tokens
            retry_after = None
            if not allowed:
                tokens_needed = cost - bucket.tokens
                retry_after = tokens_needed / refill_rate

            reset_at = now + (capacity - bucket.tokens) / refill_rate

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_at=reset_at,
                retry_after=retry_after,
                limit=capacity,
                bucket_key=key,
                message="" if allowed else f"Rate limit exceeded. Retry after {retry_after:.1f}s",
            )

    def get_tokens(self, key: str) -> float:
        """Get current token count for a key."""
        with self._lock:
            bucket = self._buckets.get(key)
            if not bucket:
                return float(self.config.burst_size)

            # Calculate current tokens
            now = time.time()
            elapsed = now - bucket.last_update
            refill_rate = self.config.requests_per_minute / 60.0

            return min(
                self.config.burst_size,
                bucket.tokens + elapsed * refill_rate,
            )

    def reset(self, key: Optional[str] = None) -> None:
        """Reset bucket for a key or all keys."""
        with self._lock:
            if key:
                self._buckets.pop(key, None)
            else:
                self._buckets.clear()


class CompositeRateLimiter:
    """
    Composite rate limiter combining multiple strategies.

    Checks sliding window for sustained rate and token bucket
    for burst handling.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize composite rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self.sliding_window = SlidingWindowRateLimiter(config)
        self.token_bucket = TokenBucketRateLimiter(config)

    def check(
        self,
        key: str,
        cost: int = 1,
    ) -> RateLimitResult:
        """
        Check rate limit using both strategies.

        Args:
            key: Rate limit key
            cost: Request cost

        Returns:
            RateLimitResult (most restrictive)
        """
        # Check burst limit first (token bucket)
        burst_result = self.token_bucket.check(key, cost)
        if not burst_result.allowed:
            return burst_result

        # Check sustained rate (sliding window)
        sustained_result = self.sliding_window.check(key, cost)
        if not sustained_result.allowed:
            # Refund token bucket since we're not allowing
            self.token_bucket._buckets[key].tokens += cost
            return sustained_result

        # Return the more restrictive result
        return RateLimitResult(
            allowed=True,
            remaining=min(burst_result.remaining, sustained_result.remaining),
            reset_at=max(burst_result.reset_at, sustained_result.reset_at),
            limit=min(burst_result.limit, sustained_result.limit),
            bucket_key=key,
        )

    def build_key(
        self,
        user_id: Optional[str] = None,
        voice_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> str:
        """
        Build a rate limit key from components.

        Args:
            user_id: User identifier
            voice_id: Voice identifier
            ip_address: Client IP address
            operation: Operation type

        Returns:
            Composite key string
        """
        parts = []

        if operation:
            parts.append(f"op:{operation}")

        if self.config.per_user and user_id:
            parts.append(f"user:{user_id}")

        if self.config.per_voice and voice_id:
            parts.append(f"voice:{voice_id}")

        if self.config.per_ip and ip_address:
            parts.append(f"ip:{ip_address}")

        return ":".join(parts) if parts else "global"

    def reset(self, key: Optional[str] = None) -> None:
        """Reset both limiters."""
        self.sliding_window.reset(key)
        self.token_bucket.reset(key)


class TieredRateLimiter:
    """
    Rate limiter with tier-based limits (free, basic, pro, enterprise).
    """

    def __init__(self):
        """Initialize tiered rate limiter."""
        self._limiters: Dict[str, CompositeRateLimiter] = {}
        self._user_tiers: Dict[str, str] = {}
        self._lock = threading.RLock()

        # Pre-create limiters for each tier
        for tier, config in RATE_LIMIT_DEFAULTS.items():
            self._limiters[tier] = CompositeRateLimiter(config)

    def set_user_tier(self, user_id: str, tier: str) -> None:
        """Set the tier for a user."""
        if tier not in RATE_LIMIT_DEFAULTS:
            raise ValueError(f"Unknown tier: {tier}")
        with self._lock:
            self._user_tiers[user_id] = tier

    def get_user_tier(self, user_id: str) -> str:
        """Get the tier for a user (default: free)."""
        return self._user_tiers.get(user_id, "free")

    def check(
        self,
        user_id: str,
        voice_id: Optional[str] = None,
        operation: Optional[str] = None,
        cost: int = 1,
    ) -> RateLimitResult:
        """
        Check rate limit for a user based on their tier.

        Args:
            user_id: User identifier
            voice_id: Optional voice identifier
            operation: Operation type
            cost: Request cost

        Returns:
            RateLimitResult
        """
        tier = self.get_user_tier(user_id)
        limiter = self._limiters[tier]

        key = limiter.build_key(
            user_id=user_id,
            voice_id=voice_id,
            operation=operation,
        )

        result = limiter.check(key, cost)

        # Add tier info to result
        result.metadata = {"tier": tier}

        return result


# Singleton instance
_limiter_instance: Optional[CompositeRateLimiter] = None


def get_rate_limiter(
    config: Optional[RateLimitConfig] = None,
) -> CompositeRateLimiter:
    """Get or create rate limiter singleton."""
    global _limiter_instance
    if _limiter_instance is None:
        _limiter_instance = CompositeRateLimiter(config=config)
    return _limiter_instance


def set_rate_limiter(limiter: CompositeRateLimiter) -> None:
    """Set the rate limiter singleton."""
    global _limiter_instance
    _limiter_instance = limiter
