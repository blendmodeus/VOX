"""
VØX SDK Configuration
---------------------

Configuration management for VØX SDK.

Supports:
    - Environment variables
    - Configuration files
    - Programmatic configuration
    - Per-request overrides

AXIØM Phase 8: Integrate - "How do the parts connect?"
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class Environment(str, Enum):
    """SDK environments."""
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    LOCAL = "local"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_rate_limit: bool = True
    retry_on_server_error: bool = True
    retry_on_timeout: bool = True


@dataclass
class TimeoutConfig:
    """Configuration for timeouts."""
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 60.0
    synthesis_timeout_seconds: float = 120.0
    streaming_timeout_seconds: float = 300.0
    enrollment_timeout_seconds: float = 60.0
    verification_timeout_seconds: float = 30.0


@dataclass
class QualityConfig:
    """Configuration for quality requirements."""
    min_quality_score: float = 0.6
    enable_quality_monitoring: bool = True
    fail_on_quality_gate: bool = False
    warn_on_low_quality: bool = True


@dataclass
class GovernanceConfig:
    """Configuration for governance behavior."""
    require_consent: bool = True
    strict_governance: bool = False
    enable_content_filter: bool = True
    enable_rate_limiting: bool = True
    enable_quota_tracking: bool = True


@dataclass
class BiometricConfig:
    """Configuration for biometric operations."""
    require_liveness: bool = True
    similarity_threshold: float = 0.75
    min_enrollment_samples: int = 3
    enable_drift_detection: bool = True


@dataclass
class VoxConfig:
    """
    Complete VØX SDK configuration.

    Can be loaded from:
        - Environment variables (VOX_* prefix)
        - Configuration dictionary
        - Configuration file
    """

    # Connection
    api_key: Optional[str] = None
    base_url: str = "http://localhost:8000"
    environment: Environment = Environment.LOCAL

    # Authentication
    token: Optional[str] = None
    token_refresh_enabled: bool = True

    # User context
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Voice defaults
    default_voice_id: str = "axiom_default"
    default_emotion_preset: Optional[str] = None
    default_speaking_rate: float = 1.0

    # Output
    default_audio_format: str = "mp3"
    default_sample_rate: int = 24000

    # Sub-configs
    retry: RetryConfig = field(default_factory=RetryConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    biometric: BiometricConfig = field(default_factory=BiometricConfig)

    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_requests: bool = False
    log_responses: bool = False

    # Advanced
    verify_ssl: bool = True
    proxy: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "VoxConfig":
        """
        Load configuration from environment variables.

        Environment variables:
            VOX_API_KEY: API key for authentication
            VOX_BASE_URL: Base URL for VØX API
            VOX_ENVIRONMENT: Environment (production, staging, development, local)
            VOX_USER_ID: Default user ID
            VOX_DEFAULT_VOICE: Default voice ID
            VOX_LOG_LEVEL: Logging level
            VOX_MAX_RETRIES: Maximum retry attempts
            VOX_CONNECT_TIMEOUT: Connection timeout in seconds
            VOX_READ_TIMEOUT: Read timeout in seconds

        Returns:
            VoxConfig instance
        """
        config = cls()

        # Connection
        if api_key := os.getenv("VOX_API_KEY"):
            config.api_key = api_key
        if base_url := os.getenv("VOX_BASE_URL"):
            config.base_url = base_url
        if env := os.getenv("VOX_ENVIRONMENT"):
            try:
                config.environment = Environment(env.lower())
            except ValueError:
                pass

        # User context
        if user_id := os.getenv("VOX_USER_ID"):
            config.user_id = user_id

        # Voice defaults
        if voice := os.getenv("VOX_DEFAULT_VOICE"):
            config.default_voice_id = voice
        if emotion := os.getenv("VOX_DEFAULT_EMOTION"):
            config.default_emotion_preset = emotion

        # Logging
        if log_level := os.getenv("VOX_LOG_LEVEL"):
            try:
                config.log_level = LogLevel(log_level.lower())
            except ValueError:
                pass

        # Retry
        if max_retries := os.getenv("VOX_MAX_RETRIES"):
            try:
                config.retry.max_retries = int(max_retries)
            except ValueError:
                pass

        # Timeouts
        if connect_timeout := os.getenv("VOX_CONNECT_TIMEOUT"):
            try:
                config.timeout.connect_timeout_seconds = float(connect_timeout)
            except ValueError:
                pass
        if read_timeout := os.getenv("VOX_READ_TIMEOUT"):
            try:
                config.timeout.read_timeout_seconds = float(read_timeout)
            except ValueError:
                pass

        # Quality
        if min_quality := os.getenv("VOX_MIN_QUALITY"):
            try:
                config.quality.min_quality_score = float(min_quality)
            except ValueError:
                pass

        # Governance
        if strict := os.getenv("VOX_STRICT_GOVERNANCE"):
            config.governance.strict_governance = strict.lower() in ("true", "1", "yes")

        # Biometric
        if liveness := os.getenv("VOX_REQUIRE_LIVENESS"):
            config.biometric.require_liveness = liveness.lower() in ("true", "1", "yes")

        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoxConfig":
        """
        Load configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            VoxConfig instance
        """
        config = cls()

        # Simple fields
        for key in ["api_key", "base_url", "token", "user_id", "session_id",
                    "default_voice_id", "default_emotion_preset", "default_speaking_rate",
                    "default_audio_format", "default_sample_rate", "verify_ssl", "proxy"]:
            if key in data:
                setattr(config, key, data[key])

        # Environment
        if "environment" in data:
            config.environment = Environment(data["environment"])

        # Log level
        if "log_level" in data:
            config.log_level = LogLevel(data["log_level"])

        # Sub-configs
        if "retry" in data:
            config.retry = RetryConfig(**data["retry"])
        if "timeout" in data:
            config.timeout = TimeoutConfig(**data["timeout"])
        if "quality" in data:
            config.quality = QualityConfig(**data["quality"])
        if "governance" in data:
            config.governance = GovernanceConfig(**data["governance"])
        if "biometric" in data:
            config.biometric = BiometricConfig(**data["biometric"])

        # Custom headers
        if "custom_headers" in data:
            config.custom_headers = data["custom_headers"]

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "api_key": "***" if self.api_key else None,  # Redact
            "base_url": self.base_url,
            "environment": self.environment.value,
            "user_id": self.user_id,
            "default_voice_id": self.default_voice_id,
            "default_emotion_preset": self.default_emotion_preset,
            "default_speaking_rate": self.default_speaking_rate,
            "default_audio_format": self.default_audio_format,
            "log_level": self.log_level.value,
            "retry": {
                "max_retries": self.retry.max_retries,
                "initial_delay_seconds": self.retry.initial_delay_seconds,
            },
            "timeout": {
                "connect_timeout_seconds": self.timeout.connect_timeout_seconds,
                "read_timeout_seconds": self.timeout.read_timeout_seconds,
            },
            "quality": {
                "min_quality_score": self.quality.min_quality_score,
                "enable_quality_monitoring": self.quality.enable_quality_monitoring,
            },
            "governance": {
                "require_consent": self.governance.require_consent,
                "strict_governance": self.governance.strict_governance,
            },
            "biometric": {
                "require_liveness": self.biometric.require_liveness,
                "similarity_threshold": self.biometric.similarity_threshold,
            },
        }

    def with_overrides(self, **kwargs) -> "VoxConfig":
        """
        Create new config with overrides.

        Args:
            **kwargs: Fields to override

        Returns:
            New VoxConfig with overrides applied
        """
        import copy
        new_config = copy.deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
        return new_config

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # API key required for non-local environments
        if self.environment != Environment.LOCAL and not self.api_key:
            errors.append("API key required for non-local environments")

        # Validate timeouts
        if self.timeout.connect_timeout_seconds <= 0:
            errors.append("Connect timeout must be positive")
        if self.timeout.read_timeout_seconds <= 0:
            errors.append("Read timeout must be positive")

        # Validate quality
        if not 0 <= self.quality.min_quality_score <= 1:
            errors.append("Min quality score must be between 0 and 1")

        # Validate biometric
        if not 0 <= self.biometric.similarity_threshold <= 1:
            errors.append("Similarity threshold must be between 0 and 1")
        if self.biometric.min_enrollment_samples < 1:
            errors.append("Min enrollment samples must be at least 1")

        # Validate retry
        if self.retry.max_retries < 0:
            errors.append("Max retries must be non-negative")
        if self.retry.initial_delay_seconds <= 0:
            errors.append("Initial delay must be positive")

        return errors


# Default configurations for different environments
DEFAULT_CONFIGS = {
    Environment.PRODUCTION: VoxConfig(
        base_url="https://api.axiom-vox.com",
        environment=Environment.PRODUCTION,
        verify_ssl=True,
        governance=GovernanceConfig(
            require_consent=True,
            strict_governance=True,
            enable_content_filter=True,
        ),
        quality=QualityConfig(
            min_quality_score=0.7,
            fail_on_quality_gate=True,
        ),
    ),
    Environment.STAGING: VoxConfig(
        base_url="https://staging-api.axiom-vox.com",
        environment=Environment.STAGING,
        verify_ssl=True,
        governance=GovernanceConfig(
            require_consent=True,
            strict_governance=False,
        ),
    ),
    Environment.DEVELOPMENT: VoxConfig(
        base_url="https://dev-api.axiom-vox.com",
        environment=Environment.DEVELOPMENT,
        verify_ssl=False,
        log_requests=True,
        log_responses=True,
    ),
    Environment.LOCAL: VoxConfig(
        base_url="http://localhost:8000",
        environment=Environment.LOCAL,
        verify_ssl=False,
        governance=GovernanceConfig(
            require_consent=False,
            strict_governance=False,
        ),
        log_requests=True,
    ),
}


def get_default_config(environment: Environment = Environment.LOCAL) -> VoxConfig:
    """Get default configuration for an environment."""
    return DEFAULT_CONFIGS.get(environment, DEFAULT_CONFIGS[Environment.LOCAL])
