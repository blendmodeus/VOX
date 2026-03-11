"""
VØX SDK Retry Logic
-------------------

Retry and backoff strategies for VØX SDK.

Features:
    - Exponential backoff with jitter
    - Automatic rate limit handling
    - Configurable retry policies
    - Async and sync support

AXIØM Phase 8: Integrate - "How do the parts connect?"
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import (
    Callable,
    TypeVar,
    Optional,
    List,
    Type,
    Union,
    Awaitable,
    Any,
)

from .config import RetryConfig
from .errors import (
    VoxError,
    RateLimitError,
    QuotaExceededError,
    ResourceError,
    NetworkError,
    SynthesisError,
    RetryStrategy,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryState:
    """State tracking for retry attempts."""
    attempt: int = 0
    total_delay: float = 0.0
    last_error: Optional[VoxError] = None
    errors: List[VoxError] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class RetryPolicy:
    """
    Configurable retry policy for VØX operations.

    Supports:
        - Exponential backoff with optional jitter
        - Rate limit aware waiting
        - Configurable retryable errors
        - Maximum retry attempts and delays
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        retryable_errors: Optional[List[Type[VoxError]]] = None,
    ):
        """
        Initialize retry policy.

        Args:
            config: Retry configuration
            retryable_errors: List of error types to retry
        """
        self.config = config or RetryConfig()
        self.retryable_errors = retryable_errors or [
            RateLimitError,
            QuotaExceededError,
            ResourceError,
            NetworkError,
            SynthesisError,
        ]

    def should_retry(self, error: VoxError, state: RetryState) -> bool:
        """
        Determine if operation should be retried.

        Args:
            error: The error that occurred
            state: Current retry state

        Returns:
            True if should retry
        """
        # Check max retries
        if state.attempt >= self.config.max_retries:
            logger.debug(f"Max retries ({self.config.max_retries}) exceeded")
            return False

        # Check if error is retryable
        if not error.is_retryable:
            logger.debug(f"Error {type(error).__name__} is not retryable")
            return False

        # Check if error type is in retryable list
        if not any(isinstance(error, t) for t in self.retryable_errors):
            logger.debug(f"Error type {type(error).__name__} not in retryable list")
            return False

        # Check specific config flags
        if isinstance(error, RateLimitError) and not self.config.retry_on_rate_limit:
            return False
        if isinstance(error, (ResourceError, SynthesisError)) and not self.config.retry_on_server_error:
            return False
        if isinstance(error, NetworkError) and not self.config.retry_on_timeout:
            return False

        return True

    def get_delay(self, error: VoxError, state: RetryState) -> float:
        """
        Calculate delay before next retry.

        Args:
            error: The error that occurred
            state: Current retry state

        Returns:
            Delay in seconds
        """
        # Use retry_after if provided (rate limits, quotas)
        if error.retry_after and error.retry_after > 0:
            delay = error.retry_after
            logger.debug(f"Using retry_after from error: {delay:.2f}s")
            return min(delay, self.config.max_delay_seconds)

        # Calculate exponential backoff
        delay = self.config.initial_delay_seconds * (
            self.config.exponential_base ** state.attempt
        )

        # Apply jitter if enabled
        if self.config.jitter:
            jitter_range = delay * 0.2  # ±20% jitter
            delay += random.uniform(-jitter_range, jitter_range)

        # Clamp to max delay
        delay = min(delay, self.config.max_delay_seconds)

        logger.debug(f"Calculated backoff delay: {delay:.2f}s (attempt {state.attempt})")
        return delay


class RetryExecutor:
    """
    Executes operations with retry logic.

    Supports both sync and async operations.
    """

    def __init__(self, policy: Optional[RetryPolicy] = None):
        """
        Initialize retry executor.

        Args:
            policy: Retry policy to use
        """
        self.policy = policy or RetryPolicy()

    async def execute_async(
        self,
        operation: Callable[[], Awaitable[T]],
        operation_name: str = "operation",
    ) -> T:
        """
        Execute async operation with retry.

        Args:
            operation: Async operation to execute
            operation_name: Name for logging

        Returns:
            Operation result

        Raises:
            VoxError: If all retries exhausted
        """
        state = RetryState()

        while True:
            try:
                result = await operation()
                if state.attempt > 0:
                    logger.info(
                        f"{operation_name} succeeded after {state.attempt} retries"
                    )
                return result

            except VoxError as error:
                state.attempt += 1
                state.last_error = error
                state.errors.append(error)

                if not self.policy.should_retry(error, state):
                    logger.warning(
                        f"{operation_name} failed after {state.attempt} attempts: {error}"
                    )
                    raise

                delay = self.policy.get_delay(error, state)
                state.total_delay += delay

                logger.info(
                    f"{operation_name} failed (attempt {state.attempt}), "
                    f"retrying in {delay:.2f}s: {error.message}"
                )

                await asyncio.sleep(delay)

    def execute_sync(
        self,
        operation: Callable[[], T],
        operation_name: str = "operation",
    ) -> T:
        """
        Execute sync operation with retry.

        Args:
            operation: Sync operation to execute
            operation_name: Name for logging

        Returns:
            Operation result

        Raises:
            VoxError: If all retries exhausted
        """
        state = RetryState()

        while True:
            try:
                result = operation()
                if state.attempt > 0:
                    logger.info(
                        f"{operation_name} succeeded after {state.attempt} retries"
                    )
                return result

            except VoxError as error:
                state.attempt += 1
                state.last_error = error
                state.errors.append(error)

                if not self.policy.should_retry(error, state):
                    logger.warning(
                        f"{operation_name} failed after {state.attempt} attempts: {error}"
                    )
                    raise

                delay = self.policy.get_delay(error, state)
                state.total_delay += delay

                logger.info(
                    f"{operation_name} failed (attempt {state.attempt}), "
                    f"retrying in {delay:.2f}s: {error.message}"
                )

                time.sleep(delay)


def with_retry(
    policy: Optional[RetryPolicy] = None,
    operation_name: Optional[str] = None,
):
    """
    Decorator to add retry logic to a function.

    Args:
        policy: Retry policy (default policy if None)
        operation_name: Name for logging (defaults to function name)

    Returns:
        Decorated function with retry logic

    Example:
        @with_retry(policy=RetryPolicy(config=RetryConfig(max_retries=5)))
        async def synthesize(text: str) -> bytes:
            ...
    """
    executor = RetryExecutor(policy)

    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await executor.execute_async(
                    lambda: func(*args, **kwargs),
                    operation_name=name,
                )
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return executor.execute_sync(
                    lambda: func(*args, **kwargs),
                    operation_name=name,
                )
            return sync_wrapper

    return decorator


class RateLimitHandler:
    """
    Specialized handler for rate limit responses.

    Provides:
        - Automatic waiting for rate limit reset
        - Proactive rate limit checking
        - Rate limit state tracking
    """

    def __init__(self):
        """Initialize rate limit handler."""
        self._rate_limit_state: dict = {}

    def record_rate_limit(
        self,
        key: str,
        limit: int,
        remaining: int,
        reset_at: float,
    ) -> None:
        """
        Record rate limit state from response headers.

        Args:
            key: Rate limit key (e.g., "user:123" or "global")
            limit: Total limit
            remaining: Remaining requests
            reset_at: Unix timestamp when limit resets
        """
        self._rate_limit_state[key] = {
            "limit": limit,
            "remaining": remaining,
            "reset_at": reset_at,
            "recorded_at": time.time(),
        }

    def get_remaining(self, key: str) -> Optional[int]:
        """Get remaining requests for a key."""
        state = self._rate_limit_state.get(key)
        if not state:
            return None

        # Check if reset has passed
        if time.time() >= state["reset_at"]:
            return state["limit"]

        return state["remaining"]

    def should_wait(self, key: str) -> tuple[bool, float]:
        """
        Check if we should wait before making a request.

        Args:
            key: Rate limit key

        Returns:
            Tuple of (should_wait, wait_seconds)
        """
        state = self._rate_limit_state.get(key)
        if not state:
            return False, 0.0

        now = time.time()

        # If reset has passed, no need to wait
        if now >= state["reset_at"]:
            return False, 0.0

        # If we have remaining requests, no need to wait
        if state["remaining"] > 0:
            return False, 0.0

        # Calculate wait time
        wait_time = state["reset_at"] - now
        return True, wait_time

    async def wait_if_needed(self, key: str) -> None:
        """
        Wait if rate limited.

        Args:
            key: Rate limit key
        """
        should_wait, wait_time = self.should_wait(key)
        if should_wait:
            logger.info(f"Rate limited, waiting {wait_time:.2f}s before request")
            await asyncio.sleep(wait_time)

    def clear(self, key: Optional[str] = None) -> None:
        """Clear rate limit state."""
        if key:
            self._rate_limit_state.pop(key, None)
        else:
            self._rate_limit_state.clear()


# Global rate limit handler
_rate_limit_handler = RateLimitHandler()


def get_rate_limit_handler() -> RateLimitHandler:
    """Get the global rate limit handler."""
    return _rate_limit_handler
