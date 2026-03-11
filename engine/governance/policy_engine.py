"""
Policy Engine
-------------

Content and usage policy enforcement for voice operations.

AXIØM Phase 7: Constrain - "What limits clarify the solution?"
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Pattern
import hashlib

from .models import (
    Policy,
    PolicyResult,
    PolicyType,
    PolicyAction,
    ContentPolicy,
    UsagePolicy,
    ViolationType,
)

logger = logging.getLogger(__name__)


# Default blocked patterns (regex)
DEFAULT_BLOCKED_PATTERNS = [
    r"(?i)\b(password|passwd|pwd)\s*[:=]\s*\S+",  # Credentials
    r"(?i)\b(api[_-]?key|secret[_-]?key)\s*[:=]\s*\S+",  # API keys
    r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN pattern
    r"\b\d{16}\b",  # Credit card pattern
]

# Default blocked topics
DEFAULT_BLOCKED_TOPICS = [
    "violence_instruction",
    "self_harm",
    "illegal_activity",
    "hate_speech",
    "explicit_content",
]


@dataclass
class ContentAnalysis:
    """Result of content analysis."""
    text_length: int
    word_count: int
    language: Optional[str] = None
    contains_pii: bool = False
    pii_types: List[str] = field(default_factory=list)
    blocked_patterns_found: List[str] = field(default_factory=list)
    profanity_detected: bool = False
    sentiment: Optional[str] = None


class ContentFilter:
    """
    Filters and validates content for synthesis.

    Checks for:
        - Blocked patterns (PII, credentials)
        - Content length limits
        - Profanity
        - Language detection
    """

    def __init__(
        self,
        policy: Optional[ContentPolicy] = None,
    ):
        """
        Initialize content filter.

        Args:
            policy: Content policy configuration
        """
        self.policy = policy or ContentPolicy()
        self._compiled_patterns: List[Pattern] = []
        self._compile_patterns()

        # Simple profanity list (extend as needed)
        self._profanity_words: Set[str] = set()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        patterns = self.policy.blocked_patterns or DEFAULT_BLOCKED_PATTERNS
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in patterns
        ]

    def analyze(self, text: str) -> ContentAnalysis:
        """
        Analyze text content.

        Args:
            text: Text to analyze

        Returns:
            ContentAnalysis
        """
        analysis = ContentAnalysis(
            text_length=len(text),
            word_count=len(text.split()),
        )

        # Check for blocked patterns
        for pattern in self._compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                analysis.blocked_patterns_found.append(pattern.pattern)
                analysis.contains_pii = True

        # Simple PII detection
        pii_checks = [
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
            (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "phone"),
            (r"\b\d{5}(-\d{4})?\b", "zipcode"),
        ]

        for pattern, pii_type in pii_checks:
            if re.search(pattern, text):
                analysis.pii_types.append(pii_type)
                if self.policy.pii_detection:
                    analysis.contains_pii = True

        # Profanity check (simple)
        words = set(text.lower().split())
        if words & self._profanity_words:
            analysis.profanity_detected = True

        return analysis

    def check(self, text: str) -> PolicyResult:
        """
        Check text against content policy.

        Args:
            text: Text to check

        Returns:
            PolicyResult
        """
        analysis = self.analyze(text)
        violations = []
        warnings = []

        # Length checks
        if analysis.text_length < self.policy.min_text_length:
            violations.append(f"Text too short: {analysis.text_length} < {self.policy.min_text_length}")

        if analysis.text_length > self.policy.max_text_length:
            violations.append(f"Text too long: {analysis.text_length} > {self.policy.max_text_length}")

        # Blocked patterns
        if analysis.blocked_patterns_found:
            violations.append(f"Blocked patterns detected: {len(analysis.blocked_patterns_found)}")

        # PII
        if analysis.contains_pii and self.policy.pii_detection:
            warnings.append(f"PII detected: {', '.join(analysis.pii_types)}")

        # Profanity
        if analysis.profanity_detected and self.policy.profanity_filter:
            warnings.append("Profanity detected")

        # Determine action
        if violations:
            action = PolicyAction.BLOCK
            allowed = False
        elif warnings:
            action = PolicyAction.WARN
            allowed = True
        else:
            action = PolicyAction.ALLOW
            allowed = True

        return PolicyResult(
            policy_id="content_filter",
            policy_type=PolicyType.CONTENT,
            action=action,
            allowed=allowed,
            violations=violations,
            warnings=warnings,
            metadata={"analysis": analysis.__dict__},
        )


class UsageValidator:
    """
    Validates usage patterns and permissions.

    Checks for:
        - Allowed operations
        - Authentication requirements
        - Consent requirements
        - Commercial use restrictions
    """

    def __init__(
        self,
        policy: Optional[UsagePolicy] = None,
    ):
        """
        Initialize usage validator.

        Args:
            policy: Usage policy configuration
        """
        self.policy = policy or UsagePolicy()

    def check(
        self,
        operation: str,
        user_id: Optional[str] = None,
        voice_id: Optional[str] = None,
        has_consent: bool = False,
        is_commercial: bool = False,
        is_third_party: bool = False,
        audio_duration_seconds: float = 0.0,
        concurrent_requests: int = 0,
    ) -> PolicyResult:
        """
        Check usage against policy.

        Args:
            operation: Operation being performed
            user_id: User identifier
            voice_id: Voice identifier
            has_consent: Whether consent is verified
            is_commercial: Whether commercial use
            is_third_party: Whether third-party use
            audio_duration_seconds: Duration of audio
            concurrent_requests: Current concurrent requests

        Returns:
            PolicyResult
        """
        violations = []
        warnings = []

        # Operation allowed
        if operation not in self.policy.allowed_operations:
            violations.append(f"Operation not allowed: {operation}")

        # Authentication
        if self.policy.require_authentication and not user_id:
            violations.append("Authentication required")

        # Consent
        if self.policy.require_consent and not has_consent:
            violations.append("Consent required")

        # Commercial use
        if is_commercial and not self.policy.allow_commercial:
            violations.append("Commercial use not allowed")

        # Third-party use
        if is_third_party and not self.policy.allow_third_party:
            violations.append("Third-party use not allowed")

        # Audio duration
        if audio_duration_seconds > self.policy.max_audio_duration_seconds:
            violations.append(
                f"Audio duration exceeds limit: {audio_duration_seconds} > "
                f"{self.policy.max_audio_duration_seconds}"
            )

        # Concurrent requests
        if concurrent_requests >= self.policy.max_concurrent_requests:
            warnings.append(
                f"At concurrent request limit: {concurrent_requests}"
            )

        # Determine action
        if violations:
            action = PolicyAction.BLOCK
            allowed = False
        elif warnings:
            action = PolicyAction.WARN
            allowed = True
        else:
            action = PolicyAction.ALLOW
            allowed = True

        return PolicyResult(
            policy_id="usage_validator",
            policy_type=PolicyType.USAGE,
            action=action,
            allowed=allowed,
            violations=violations,
            warnings=warnings,
        )


class PolicyEngine:
    """
    Central policy enforcement engine.

    Coordinates content filtering, usage validation,
    and custom policy evaluation.
    """

    def __init__(
        self,
        content_policy: Optional[ContentPolicy] = None,
        usage_policy: Optional[UsagePolicy] = None,
    ):
        """
        Initialize policy engine.

        Args:
            content_policy: Content policy configuration
            usage_policy: Usage policy configuration
        """
        self.content_filter = ContentFilter(content_policy)
        self.usage_validator = UsageValidator(usage_policy)
        self._custom_policies: Dict[str, Policy] = {}

    def register_policy(self, policy: Policy) -> None:
        """Register a custom policy."""
        self._custom_policies[policy.policy_id] = policy
        logger.info(f"Registered policy: {policy.policy_id}")

    def unregister_policy(self, policy_id: str) -> bool:
        """Unregister a custom policy."""
        if policy_id in self._custom_policies:
            del self._custom_policies[policy_id]
            return True
        return False

    def evaluate_content(self, text: str) -> PolicyResult:
        """Evaluate text against content policy."""
        return self.content_filter.check(text)

    def evaluate_usage(
        self,
        operation: str,
        **kwargs,
    ) -> PolicyResult:
        """Evaluate usage against usage policy."""
        return self.usage_validator.check(operation, **kwargs)

    def evaluate_all(
        self,
        text: str,
        operation: str,
        **kwargs,
    ) -> List[PolicyResult]:
        """
        Evaluate against all policies.

        Args:
            text: Text content
            operation: Operation being performed
            **kwargs: Additional context

        Returns:
            List of PolicyResults
        """
        results = []

        # Content policy
        content_result = self.evaluate_content(text)
        results.append(content_result)

        # Usage policy
        usage_result = self.evaluate_usage(operation, **kwargs)
        results.append(usage_result)

        # Custom policies
        for policy in self._custom_policies.values():
            if not policy.enabled:
                continue

            result = self._evaluate_custom_policy(
                policy,
                text=text,
                operation=operation,
                **kwargs,
            )
            results.append(result)

        return results

    def _evaluate_custom_policy(
        self,
        policy: Policy,
        **context,
    ) -> PolicyResult:
        """Evaluate a custom policy."""
        violations = []
        warnings = []

        # Simple condition matching
        conditions = policy.conditions

        # Text length condition
        if "max_text_length" in conditions:
            text = context.get("text", "")
            if len(text) > conditions["max_text_length"]:
                violations.append(f"Text exceeds custom limit")

        # Time-based conditions
        if "allowed_hours" in conditions:
            hour = time.localtime().tm_hour
            if hour not in conditions["allowed_hours"]:
                violations.append("Outside allowed hours")

        # User-based conditions
        if "blocked_users" in conditions:
            user_id = context.get("user_id")
            if user_id in conditions["blocked_users"]:
                violations.append("User blocked")

        # Voice-based conditions
        if "blocked_voices" in conditions:
            voice_id = context.get("voice_id")
            if voice_id in conditions["blocked_voices"]:
                violations.append("Voice blocked")

        # Determine action
        if violations:
            action = policy.action
            allowed = action not in [PolicyAction.BLOCK, PolicyAction.ESCALATE]
        else:
            action = PolicyAction.ALLOW
            allowed = True

        return PolicyResult(
            policy_id=policy.policy_id,
            policy_type=policy.policy_type,
            action=action,
            allowed=allowed,
            violations=violations,
            warnings=warnings,
        )

    def is_allowed(
        self,
        text: str,
        operation: str,
        **kwargs,
    ) -> bool:
        """
        Quick check if request is allowed.

        Args:
            text: Text content
            operation: Operation
            **kwargs: Additional context

        Returns:
            True if all policies allow
        """
        results = self.evaluate_all(text, operation, **kwargs)
        return all(r.allowed for r in results)

    def get_violations(
        self,
        text: str,
        operation: str,
        **kwargs,
    ) -> List[str]:
        """
        Get all policy violations.

        Args:
            text: Text content
            operation: Operation
            **kwargs: Additional context

        Returns:
            List of violation messages
        """
        results = self.evaluate_all(text, operation, **kwargs)
        violations = []
        for r in results:
            violations.extend(r.violations)
        return violations

    def get_action(
        self,
        text: str,
        operation: str,
        **kwargs,
    ) -> PolicyAction:
        """
        Get the most restrictive action.

        Args:
            text: Text content
            operation: Operation
            **kwargs: Additional context

        Returns:
            Most restrictive PolicyAction
        """
        results = self.evaluate_all(text, operation, **kwargs)

        # Priority order: BLOCK > ESCALATE > THROTTLE > QUEUE > WARN > ALLOW
        action_priority = {
            PolicyAction.BLOCK: 5,
            PolicyAction.ESCALATE: 4,
            PolicyAction.THROTTLE: 3,
            PolicyAction.QUEUE: 2,
            PolicyAction.WARN: 1,
            PolicyAction.ALLOW: 0,
        }

        return max(
            (r.action for r in results),
            key=lambda a: action_priority.get(a, 0),
        )


# Singleton instance
_engine_instance: Optional[PolicyEngine] = None


def get_policy_engine(
    content_policy: Optional[ContentPolicy] = None,
    usage_policy: Optional[UsagePolicy] = None,
) -> PolicyEngine:
    """Get or create policy engine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PolicyEngine(
            content_policy=content_policy,
            usage_policy=usage_policy,
        )
    return _engine_instance


def set_policy_engine(engine: PolicyEngine) -> None:
    """Set the policy engine singleton."""
    global _engine_instance
    _engine_instance = engine
