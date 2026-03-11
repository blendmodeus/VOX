"""
VØX SDK Tests
-------------

Comprehensive tests for the VØX Client SDK.

AXIØM Phase 8: Integrate - "How do the parts connect?"
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

# ============================================================================
# Error Tests
# ============================================================================


class TestErrors:
    """Tests for SDK error hierarchy."""

    def test_vox_error_base(self):
        """Test VoxError base class."""
        from axiom_vox.sdk.errors import VoxError, ErrorCategory

        error = VoxError("Test error")
        assert error.message == "Test error"
        assert error.category == ErrorCategory.UNKNOWN
        assert error.is_retryable is False
        assert error.retry_after is None

    def test_vox_error_with_context(self):
        """Test VoxError with context."""
        from axiom_vox.sdk.errors import VoxError, ErrorContext

        context = ErrorContext(
            request_id="req_123",
            operation="synthesize",
            voice_id="warm",
        )
        error = VoxError("Test error", context=context)
        assert error.context.request_id == "req_123"
        assert error.context.operation == "synthesize"

    def test_rate_limit_error_retryable(self):
        """Test RateLimitError is retryable."""
        import time
        from axiom_vox.sdk.errors import RateLimitError

        error = RateLimitError(
            limit=100,
            remaining=0,
            reset_at=time.time() + 30,
            retry_after=30.0
        )
        assert error.is_retryable is True
        assert error.retry_after == 30.0

    def test_authentication_error_not_retryable(self):
        """Test AuthenticationError is not retryable."""
        from axiom_vox.sdk.errors import AuthenticationError

        error = AuthenticationError("Invalid API key")
        assert error.is_retryable is False

    def test_synthesis_error_retryable(self):
        """Test SynthesisError is retryable."""
        from axiom_vox.sdk.errors import SynthesisError

        error = SynthesisError("Synthesis failed", retryable=True)
        assert error.is_retryable is True

    def test_from_http_response_429(self):
        """Test creating error from 429 response."""
        import time
        from axiom_vox.sdk.errors import from_http_response, RateLimitError

        error = from_http_response(
            status_code=429,
            response_body={
                "error": "RateLimitError",
                "message": "Rate limited",
                "details": {
                    "limit": 100,
                    "remaining": 0,
                    "reset_at": time.time() + 30,
                    "retry_after": 30
                }
            },
        )
        assert isinstance(error, RateLimitError)
        assert error.retry_after == 30

    def test_from_http_response_401(self):
        """Test creating error from 401 response."""
        from axiom_vox.sdk.errors import from_http_response, AuthenticationError

        error = from_http_response(
            status_code=401,
            response_body={"message": "Invalid token"},
        )
        assert isinstance(error, AuthenticationError)

    def test_from_http_response_500(self):
        """Test creating error from 500 response."""
        from axiom_vox.sdk.errors import from_http_response, VoxError

        error = from_http_response(
            status_code=500,
            response_body={"message": "Internal error"},
        )
        # 500 returns base VoxError per the current implementation
        assert isinstance(error, VoxError)


# ============================================================================
# Config Tests
# ============================================================================


class TestConfig:
    """Tests for SDK configuration."""

    def test_default_config(self):
        """Test default configuration."""
        from axiom_vox.sdk.config import VoxConfig, Environment

        config = VoxConfig()
        assert config.base_url == "http://localhost:8000"
        assert config.environment == Environment.LOCAL
        assert config.default_voice_id == "axiom_default"

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        import os
        from axiom_vox.sdk.config import VoxConfig

        with patch.dict(os.environ, {
            "VOX_API_KEY": "test_key",
            "VOX_BASE_URL": "https://api.example.com",
            "VOX_USER_ID": "user_123",
        }):
            config = VoxConfig.from_env()
            assert config.api_key == "test_key"
            assert config.base_url == "https://api.example.com"
            assert config.user_id == "user_123"

    def test_config_from_dict(self):
        """Test configuration from dictionary."""
        from axiom_vox.sdk.config import VoxConfig, Environment

        data = {
            "api_key": "test_key",
            "environment": "production",
            "default_voice_id": "warm",
            "retry": {
                "max_retries": 5,
            },
        }
        config = VoxConfig.from_dict(data)
        assert config.api_key == "test_key"
        assert config.environment == Environment.PRODUCTION
        assert config.default_voice_id == "warm"
        assert config.retry.max_retries == 5

    def test_config_to_dict_redacts_api_key(self):
        """Test that to_dict redacts API key."""
        from axiom_vox.sdk.config import VoxConfig

        config = VoxConfig(api_key="secret_key")
        data = config.to_dict()
        assert data["api_key"] == "***"

    def test_config_with_overrides(self):
        """Test config with overrides."""
        from axiom_vox.sdk.config import VoxConfig

        config = VoxConfig(api_key="original")
        new_config = config.with_overrides(api_key="new_key")
        assert config.api_key == "original"
        assert new_config.api_key == "new_key"

    def test_config_validation(self):
        """Test config validation."""
        from axiom_vox.sdk.config import VoxConfig, Environment

        # Valid local config (no API key needed)
        config = VoxConfig(environment=Environment.LOCAL)
        errors = config.validate()
        assert len(errors) == 0

        # Invalid production config (no API key)
        config = VoxConfig(environment=Environment.PRODUCTION)
        errors = config.validate()
        assert "API key required" in errors[0]

    def test_default_configs_for_environments(self):
        """Test default configs for each environment."""
        from axiom_vox.sdk.config import get_default_config, Environment

        prod = get_default_config(Environment.PRODUCTION)
        assert prod.governance.strict_governance is True
        assert prod.verify_ssl is True

        local = get_default_config(Environment.LOCAL)
        assert local.governance.require_consent is False
        assert local.verify_ssl is False


# ============================================================================
# Retry Tests
# ============================================================================


class TestRetry:
    """Tests for retry logic."""

    def test_retry_state_tracking(self):
        """Test retry state tracking."""
        from axiom_vox.sdk.retry import RetryState

        state = RetryState()
        assert state.attempt == 0
        assert state.total_delay == 0.0
        assert state.errors == []

    def test_retry_policy_should_retry(self):
        """Test retry policy decision."""
        import time
        from axiom_vox.sdk.retry import RetryPolicy, RetryState
        from axiom_vox.sdk.errors import RateLimitError, AuthenticationError
        from axiom_vox.sdk.config import RetryConfig

        policy = RetryPolicy(config=RetryConfig(max_retries=3))
        state = RetryState(attempt=1)

        # RateLimitError should retry
        rate_error = RateLimitError(
            limit=100, remaining=0, reset_at=time.time() + 5, retry_after=5.0
        )
        assert policy.should_retry(rate_error, state) is True

        # AuthenticationError should not retry
        auth_error = AuthenticationError("Invalid key")
        assert policy.should_retry(auth_error, state) is False

    def test_retry_policy_max_retries(self):
        """Test retry policy respects max retries."""
        import time
        from axiom_vox.sdk.retry import RetryPolicy, RetryState
        from axiom_vox.sdk.errors import RateLimitError
        from axiom_vox.sdk.config import RetryConfig

        policy = RetryPolicy(config=RetryConfig(max_retries=3))
        error = RateLimitError(
            limit=100, remaining=0, reset_at=time.time() + 5, retry_after=5.0
        )

        # Should retry at attempt 2
        state = RetryState(attempt=2)
        assert policy.should_retry(error, state) is True

        # Should not retry at attempt 3
        state = RetryState(attempt=3)
        assert policy.should_retry(error, state) is False

    def test_retry_policy_delay_calculation(self):
        """Test retry delay calculation."""
        import time
        from axiom_vox.sdk.retry import RetryPolicy, RetryState
        from axiom_vox.sdk.errors import NetworkError
        from axiom_vox.sdk.config import RetryConfig

        policy = RetryPolicy(config=RetryConfig(
            initial_delay_seconds=1.0,
            exponential_base=2.0,
            jitter=False,
        ))
        # Use NetworkError which doesn't have retry_after
        error = NetworkError("Connection failed")

        # Delay should increase exponentially
        assert policy.get_delay(error, RetryState(attempt=0)) == 1.0
        assert policy.get_delay(error, RetryState(attempt=1)) == 2.0
        assert policy.get_delay(error, RetryState(attempt=2)) == 4.0

    def test_retry_policy_uses_retry_after(self):
        """Test retry policy uses retry_after from error."""
        import time
        from axiom_vox.sdk.retry import RetryPolicy, RetryState
        from axiom_vox.sdk.errors import RateLimitError

        policy = RetryPolicy()
        error = RateLimitError(
            limit=100, remaining=0, reset_at=time.time() + 30, retry_after=30.0
        )
        state = RetryState(attempt=0)

        delay = policy.get_delay(error, state)
        assert delay == 30.0

    @pytest.mark.asyncio
    async def test_retry_executor_async(self):
        """Test async retry execution."""
        import time
        from axiom_vox.sdk.retry import RetryExecutor
        from axiom_vox.sdk.errors import RateLimitError
        from axiom_vox.sdk.config import RetryConfig

        executor = RetryExecutor()

        # Operation that succeeds on third try
        call_count = 0

        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError(
                    limit=100, remaining=0, reset_at=time.time() + 0.01, retry_after=0.01
                )
            return "success"

        result = await executor.execute_async(flaky_operation, "test_op")
        assert result == "success"
        assert call_count == 3

    def test_with_retry_decorator(self):
        """Test @with_retry decorator."""
        import time
        from axiom_vox.sdk.retry import with_retry
        from axiom_vox.sdk.errors import RateLimitError

        call_count = 0

        @with_retry(operation_name="test")
        def flaky_sync():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError(
                    limit=100, remaining=0, reset_at=time.time() + 0.01, retry_after=0.01
                )
            return "success"

        result = flaky_sync()
        assert result == "success"
        assert call_count == 2

    def test_rate_limit_handler(self):
        """Test rate limit handler state tracking."""
        from axiom_vox.sdk.retry import RateLimitHandler

        handler = RateLimitHandler()
        now = time.time()

        # Record rate limit
        handler.record_rate_limit(
            key="user:123",
            limit=100,
            remaining=0,
            reset_at=now + 60,
        )

        # Should wait
        should_wait, wait_time = handler.should_wait("user:123")
        assert should_wait is True
        assert 59 < wait_time <= 60

        # After reset
        handler.record_rate_limit(
            key="user:456",
            limit=100,
            remaining=50,
            reset_at=now + 60,
        )
        should_wait, _ = handler.should_wait("user:456")
        assert should_wait is False


# ============================================================================
# Session Tests
# ============================================================================


class TestSession:
    """Tests for session management."""

    @pytest.mark.asyncio
    async def test_session_lifecycle(self):
        """Test session lifecycle."""
        from axiom_vox.sdk.session import VoxSession, SessionState

        async with VoxSession() as session:
            assert session.state == SessionState.ACTIVE
            assert session.session_id.startswith("sess_")

        assert session.state == SessionState.CLOSED

    @pytest.mark.asyncio
    async def test_session_metrics(self):
        """Test session metrics tracking."""
        from axiom_vox.sdk.session import VoxSession

        session = VoxSession()
        await session._initialize()

        try:
            # Create and complete some requests
            ctx1 = session.create_request_context("synthesize", voice_id="warm")
            session.complete_request(ctx1, success=True, bytes_generated=1000, audio_seconds=5.0)

            ctx2 = session.create_request_context("synthesize", voice_id="cool")
            session.complete_request(ctx2, success=False, error="Test error")

            metrics = session.get_metrics()
            assert metrics.requests_made == 2
            assert metrics.requests_succeeded == 1
            assert metrics.requests_failed == 1
            assert metrics.bytes_synthesized == 1000
            assert metrics.audio_seconds_generated == 5.0
            assert len(metrics.errors) == 1

        finally:
            await session._cleanup()

    @pytest.mark.asyncio
    async def test_session_request_context_manager(self):
        """Test session request context manager."""
        from axiom_vox.sdk.session import VoxSession

        async with VoxSession() as session:
            async with session.request("synthesize", voice_id="warm") as ctx:
                ctx.metadata["test"] = "value"

            assert session.metrics.requests_succeeded == 1

    @pytest.mark.asyncio
    async def test_session_pool(self):
        """Test session pool."""
        from axiom_vox.sdk.session import SessionPool

        pool = SessionPool(max_sessions=2)

        try:
            session1 = await pool.get_session()
            session2 = await pool.get_session()

            assert len(pool._sessions) == 2

            metrics = pool.get_pool_metrics()
            assert metrics["total_sessions"] == 2
            assert metrics["active_sessions"] == 2

        finally:
            await pool.close_all()
            assert len(pool._sessions) == 0


# ============================================================================
# Workflow Tests
# ============================================================================


class TestWorkflows:
    """Tests for workflow helpers."""

    @pytest.mark.asyncio
    async def test_synthesize_batch(self):
        """Test batch synthesis."""
        from axiom_vox.sdk.workflows import synthesize_batch, SynthesisResult

        # Mock client
        client = AsyncMock()
        client.synthesize = AsyncMock(return_value=SynthesisResult(
            audio=b"audio",
            voice_id="warm",
            text="test",
            duration_seconds=1.0,
        ))

        items = [
            {"text": "Hello", "voice_id": "warm"},
            {"text": "World", "voice_id": "cool"},
        ]

        results = await synthesize_batch(client, items)
        assert len(results) == 2
        assert client.synthesize.call_count == 2

    @pytest.mark.asyncio
    async def test_synthesize_with_quality_check(self):
        """Test synthesis with quality check."""
        from axiom_vox.sdk.workflows import synthesize_with_quality_check, SynthesisResult

        # Mock client that returns low quality first, then good quality
        call_count = 0

        async def mock_synthesize(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            quality = 0.4 if call_count < 2 else 0.8
            return SynthesisResult(
                audio=b"audio",
                voice_id="warm",
                text="test",
                quality_score=quality,
            )

        client = AsyncMock()
        client.synthesize = mock_synthesize

        result = await synthesize_with_quality_check(
            client, "Hello", voice_id="warm", min_quality=0.6
        )
        assert result.quality_score == 0.8
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_synthesize_dialogue(self):
        """Test dialogue synthesis."""
        from axiom_vox.sdk.workflows import (
            synthesize_dialogue, DialogueLine, SynthesisResult
        )

        # Mock client
        client = AsyncMock()
        client.synthesize = AsyncMock(return_value=SynthesisResult(
            audio=b"audio",
            voice_id="warm",
            text="test",
            duration_seconds=1.0,
        ))

        lines = [
            DialogueLine(text="Hello", character="Alice", voice_id="warm"),
            DialogueLine(text="Hi there", character="Bob", voice_id="cool"),
        ]

        result = await synthesize_dialogue(client, lines)
        assert len(result.voices_used) == 2
        assert result.total_duration_seconds > 0
        assert len(result.segments) == 2

    def test_workflow_builder(self):
        """Test workflow builder."""
        from axiom_vox.sdk.workflows import WorkflowBuilder

        workflow = (
            WorkflowBuilder("test_workflow")
            .add_step("step1", "synthesize", text="Hello")
            .add_step("step2", "synthesize", text="World")
            .on_failure("step1", "step2")
            .build()
        )

        assert workflow.name == "test_workflow"
        assert len(workflow.steps) == 2
        assert "step1" in workflow.error_handlers


# ============================================================================
# Client Tests
# ============================================================================


class TestClient:
    """Tests for VoxClient."""

    def test_client_initialization(self):
        """Test client initialization."""
        from axiom_vox.sdk.client import VoxClient
        from axiom_vox.sdk.config import VoxConfig

        config = VoxConfig(api_key="test_key")
        client = VoxClient(config=config)

        assert client.config.api_key == "test_key"

    def test_client_from_params(self):
        """Test client initialization from parameters."""
        from axiom_vox.sdk.client import VoxClient

        client = VoxClient(
            api_key="test_key",
            base_url="https://api.example.com",
        )

        assert client.config.api_key == "test_key"
        assert client.config.base_url == "https://api.example.com"

    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test client as context manager."""
        from axiom_vox.sdk.client import VoxClient

        async with VoxClient() as client:
            assert client._session is not None

        # Session should be closed
        # (We can't directly check this without internal access)

    @pytest.mark.asyncio
    async def test_client_synthesize_local(self):
        """Test local synthesis."""
        from axiom_vox.sdk.client import VoxClient

        # This will use local fallback
        async with VoxClient(base_url="http://localhost:9999") as client:
            # Mock the local fallback
            with patch('axiom_vox.sdk.client.synthesize') as mock_synth:
                mock_synth.return_value = MagicMock(
                    audio=b"test_audio",
                    duration=1.0,
                    sample_rate=24000,
                )

                result = await client.synthesize("Hello world")
                assert result.audio == b"test_audio"
                mock_synth.assert_called_once()

    def test_client_version(self):
        """Test client version."""
        from axiom_vox.sdk.client import VoxClient

        client = VoxClient()
        version = client.version
        assert version == "0.13.0"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for SDK components."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full SDK workflow."""
        from axiom_vox.sdk import (
            VoxClient,
            VoxConfig,
            RetryConfig,
            Environment,
        )

        config = VoxConfig(
            environment=Environment.LOCAL,
            retry=RetryConfig(max_retries=1),
        )

        async with VoxClient(config=config) as client:
            # Client should be initialized
            assert client._session is not None

    def test_imports(self):
        """Test all SDK imports work."""
        from axiom_vox.sdk import (
            # Main client
            VoxClient,
            # Errors
            VoxError,
            AuthenticationError,
            RateLimitError,
            SynthesisError,
            # Config
            VoxConfig,
            RetryConfig,
            Environment,
            # Retry
            RetryPolicy,
            with_retry,
            # Session
            VoxSession,
            SessionPool,
            # Workflows
            WorkflowBuilder,
            synthesize_batch,
        )

        # All imports should work
        assert VoxClient is not None
        assert VoxError is not None
        assert VoxConfig is not None
        assert RetryPolicy is not None
        assert VoxSession is not None
        assert WorkflowBuilder is not None

    def test_main_module_sdk_exports(self):
        """Test SDK exports from main module."""
        from axiom_vox import (
            VoxClient,
            VoxConfig,
            VoxSession,
            __version__,
        )

        assert VoxClient is not None
        assert VoxConfig is not None
        assert VoxSession is not None
        assert __version__ == "0.13.0"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
