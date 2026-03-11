"""
Quota Manager
-------------

Usage quota tracking and enforcement for voice operations.

AXIØM Phase 7: Constrain - "What limits clarify the solution?"
"""

import logging
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import json

from .models import (
    QuotaConfig,
    QuotaStatus,
    QuotaUsage,
    QuotaPeriod,
    ViolationType,
    QUOTA_DEFAULTS,
)

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """Raised when a quota is exceeded."""

    def __init__(
        self,
        quota_name: str,
        limit: int,
        used: int,
        resets_at: float,
    ):
        self.quota_name = quota_name
        self.limit = limit
        self.used = used
        self.resets_at = resets_at
        super().__init__(
            f"Quota exceeded: {quota_name} ({used}/{limit}), "
            f"resets in {resets_at - time.time():.0f}s"
        )


def _get_period_boundaries(period: QuotaPeriod) -> Tuple[float, float]:
    """Get start and end times for a period."""
    now = time.time()
    local_time = time.localtime(now)

    if period == QuotaPeriod.MINUTE:
        period_start = now - (now % 60)
        period_end = period_start + 60
    elif period == QuotaPeriod.HOUR:
        period_start = now - (now % 3600)
        period_end = period_start + 3600
    elif period == QuotaPeriod.DAY:
        # Start of day in local time
        start_of_day = time.mktime(time.struct_time((
            local_time.tm_year, local_time.tm_mon, local_time.tm_mday,
            0, 0, 0, 0, 0, local_time.tm_isdst
        )))
        period_start = start_of_day
        period_end = period_start + 86400
    elif period == QuotaPeriod.WEEK:
        # Start of week (Monday)
        days_since_monday = local_time.tm_wday
        start_of_week = time.mktime(time.struct_time((
            local_time.tm_year, local_time.tm_mon,
            local_time.tm_mday - days_since_monday,
            0, 0, 0, 0, 0, local_time.tm_isdst
        )))
        period_start = start_of_week
        period_end = period_start + 7 * 86400
    elif period == QuotaPeriod.MONTH:
        # Start of month
        start_of_month = time.mktime(time.struct_time((
            local_time.tm_year, local_time.tm_mon, 1,
            0, 0, 0, 0, 0, local_time.tm_isdst
        )))
        period_start = start_of_month
        # Approximate month end
        if local_time.tm_mon == 12:
            next_month = time.mktime(time.struct_time((
                local_time.tm_year + 1, 1, 1,
                0, 0, 0, 0, 0, local_time.tm_isdst
            )))
        else:
            next_month = time.mktime(time.struct_time((
                local_time.tm_year, local_time.tm_mon + 1, 1,
                0, 0, 0, 0, 0, local_time.tm_isdst
            )))
        period_end = next_month
    else:
        period_start = now
        period_end = now + 60

    return period_start, period_end


class QuotaTracker:
    """
    Tracks usage for a single quota type.
    """

    def __init__(
        self,
        quota_name: str,
        limit: int,
        period: QuotaPeriod,
        warning_threshold: float = 0.8,
    ):
        """
        Initialize quota tracker.

        Args:
            quota_name: Name of the quota
            limit: Maximum usage allowed
            period: Time period for the quota
            warning_threshold: Percentage at which to warn
        """
        self.quota_name = quota_name
        self.limit = limit
        self.period = period
        self.warning_threshold = warning_threshold

        self._usage: Dict[str, QuotaUsage] = {}
        self._lock = threading.RLock()

    def _get_key(self, user_id: str, voice_id: Optional[str] = None) -> str:
        """Build usage key."""
        if voice_id:
            return f"{user_id}:{voice_id}"
        return user_id

    def _get_or_create_usage(
        self,
        user_id: str,
        voice_id: Optional[str] = None,
    ) -> QuotaUsage:
        """Get or create usage record."""
        key = self._get_key(user_id, voice_id)
        period_start, _ = _get_period_boundaries(self.period)

        with self._lock:
            usage = self._usage.get(key)

            # Check if we need to reset (new period)
            if usage is None or usage.period_start < period_start:
                usage = QuotaUsage(
                    user_id=user_id,
                    voice_id=voice_id,
                    quota_name=self.quota_name,
                    period=self.period,
                    period_start=period_start,
                    count=0,
                )
                self._usage[key] = usage

            return usage

    def check(
        self,
        user_id: str,
        voice_id: Optional[str] = None,
        amount: int = 1,
    ) -> QuotaStatus:
        """
        Check quota status without consuming.

        Args:
            user_id: User identifier
            voice_id: Optional voice identifier
            amount: Amount to check

        Returns:
            QuotaStatus
        """
        usage = self._get_or_create_usage(user_id, voice_id)
        _, period_end = _get_period_boundaries(self.period)

        remaining = max(0, self.limit - usage.count)
        exceeded = usage.count + amount > self.limit
        warning = usage.count >= self.limit * self.warning_threshold

        return QuotaStatus(
            quota_name=self.quota_name,
            period=self.period,
            limit=self.limit,
            used=usage.count,
            remaining=remaining,
            resets_at=period_end,
            exceeded=exceeded,
            warning_threshold_reached=warning,
        )

    def consume(
        self,
        user_id: str,
        voice_id: Optional[str] = None,
        amount: int = 1,
        allow_exceed: bool = False,
    ) -> QuotaStatus:
        """
        Consume quota.

        Args:
            user_id: User identifier
            voice_id: Optional voice identifier
            amount: Amount to consume
            allow_exceed: Allow exceeding (just warn)

        Returns:
            QuotaStatus

        Raises:
            QuotaExceededError: If quota exceeded and not allowed
        """
        with self._lock:
            status = self.check(user_id, voice_id, amount)

            if status.exceeded and not allow_exceed:
                raise QuotaExceededError(
                    quota_name=self.quota_name,
                    limit=self.limit,
                    used=status.used,
                    resets_at=status.resets_at,
                )

            # Consume
            usage = self._get_or_create_usage(user_id, voice_id)
            usage.count += amount
            usage.last_updated = time.time()

            # Return updated status
            return self.check(user_id, voice_id)

    def get_usage(
        self,
        user_id: str,
        voice_id: Optional[str] = None,
    ) -> int:
        """Get current usage count."""
        usage = self._get_or_create_usage(user_id, voice_id)
        return usage.count

    def reset(
        self,
        user_id: Optional[str] = None,
        voice_id: Optional[str] = None,
    ) -> None:
        """Reset usage."""
        with self._lock:
            if user_id is None:
                self._usage.clear()
            else:
                key = self._get_key(user_id, voice_id)
                self._usage.pop(key, None)


class QuotaManager:
    """
    Manages all quotas for voice operations.

    Tracks:
        - Synthesis requests per period
        - Characters processed
        - Audio duration generated
        - Voice slots used
        - Enrollments
        - Verifications
    """

    def __init__(
        self,
        config: Optional[QuotaConfig] = None,
        db=None,
    ):
        """
        Initialize quota manager.

        Args:
            config: Quota configuration
            db: Optional database for persistence
        """
        self.config = config or QuotaConfig()
        self._db = db
        self._trackers: Dict[str, QuotaTracker] = {}
        self._user_tiers: Dict[str, str] = {}
        self._lock = threading.RLock()

        self._init_trackers()

    def _init_trackers(self) -> None:
        """Initialize quota trackers from config."""
        # Synthesis quotas
        self._trackers["synthesis_minute"] = QuotaTracker(
            "synthesis_minute",
            self.config.synthesis_per_minute,
            QuotaPeriod.MINUTE,
        )
        self._trackers["synthesis_hour"] = QuotaTracker(
            "synthesis_hour",
            self.config.synthesis_per_hour,
            QuotaPeriod.HOUR,
        )
        self._trackers["synthesis_day"] = QuotaTracker(
            "synthesis_day",
            self.config.synthesis_per_day,
            QuotaPeriod.DAY,
        )

        # Character quotas
        self._trackers["characters_minute"] = QuotaTracker(
            "characters_minute",
            self.config.characters_per_minute,
            QuotaPeriod.MINUTE,
        )
        self._trackers["characters_hour"] = QuotaTracker(
            "characters_hour",
            self.config.characters_per_hour,
            QuotaPeriod.HOUR,
        )
        self._trackers["characters_day"] = QuotaTracker(
            "characters_day",
            self.config.characters_per_day,
            QuotaPeriod.DAY,
        )

        # Audio quotas
        self._trackers["audio_hour"] = QuotaTracker(
            "audio_hour",
            self.config.audio_seconds_per_hour,
            QuotaPeriod.HOUR,
        )
        self._trackers["audio_day"] = QuotaTracker(
            "audio_day",
            self.config.audio_seconds_per_day,
            QuotaPeriod.DAY,
        )

        # Biometric quotas
        self._trackers["enrollments_day"] = QuotaTracker(
            "enrollments_day",
            self.config.enrollments_per_day,
            QuotaPeriod.DAY,
        )
        self._trackers["verifications_hour"] = QuotaTracker(
            "verifications_hour",
            self.config.verifications_per_hour,
            QuotaPeriod.HOUR,
        )

    def set_user_tier(self, user_id: str, tier: str) -> None:
        """
        Set quota tier for a user.

        Args:
            user_id: User identifier
            tier: Tier name (free, basic, pro, enterprise)
        """
        if tier not in QUOTA_DEFAULTS:
            raise ValueError(f"Unknown tier: {tier}")

        with self._lock:
            self._user_tiers[user_id] = tier

            # Update trackers with tier limits
            tier_config = QUOTA_DEFAULTS[tier]
            self._update_user_limits(user_id, tier_config)

    def _update_user_limits(
        self,
        user_id: str,
        config: QuotaConfig,
    ) -> None:
        """Update limits for a user based on tier."""
        # This would store per-user limits
        # For now, trackers use default config limits
        pass

    def get_user_tier(self, user_id: str) -> str:
        """Get tier for a user."""
        return self._user_tiers.get(user_id, "free")

    def check_synthesis(
        self,
        user_id: str,
        text_length: int,
        audio_duration_seconds: float = 0.0,
    ) -> Dict[str, QuotaStatus]:
        """
        Check all synthesis-related quotas.

        Args:
            user_id: User identifier
            text_length: Length of text
            audio_duration_seconds: Expected audio duration

        Returns:
            Dict of quota statuses
        """
        statuses = {}

        # Synthesis count quotas
        for period in ["minute", "hour", "day"]:
            tracker = self._trackers[f"synthesis_{period}"]
            statuses[f"synthesis_{period}"] = tracker.check(user_id)

        # Character quotas
        for period in ["minute", "hour", "day"]:
            tracker = self._trackers[f"characters_{period}"]
            statuses[f"characters_{period}"] = tracker.check(
                user_id, amount=text_length
            )

        # Audio quotas
        audio_seconds = int(audio_duration_seconds)
        if audio_seconds > 0:
            for period in ["hour", "day"]:
                tracker = self._trackers[f"audio_{period}"]
                statuses[f"audio_{period}"] = tracker.check(
                    user_id, amount=audio_seconds
                )

        return statuses

    def consume_synthesis(
        self,
        user_id: str,
        text_length: int,
        audio_duration_seconds: float,
    ) -> Dict[str, QuotaStatus]:
        """
        Consume synthesis quotas.

        Args:
            user_id: User identifier
            text_length: Length of text
            audio_duration_seconds: Audio duration generated

        Returns:
            Dict of updated quota statuses
        """
        statuses = {}

        # Synthesis count
        for period in ["minute", "hour", "day"]:
            tracker = self._trackers[f"synthesis_{period}"]
            statuses[f"synthesis_{period}"] = tracker.consume(user_id)

        # Characters
        for period in ["minute", "hour", "day"]:
            tracker = self._trackers[f"characters_{period}"]
            statuses[f"characters_{period}"] = tracker.consume(
                user_id, amount=text_length
            )

        # Audio
        audio_seconds = int(audio_duration_seconds)
        if audio_seconds > 0:
            for period in ["hour", "day"]:
                tracker = self._trackers[f"audio_{period}"]
                statuses[f"audio_{period}"] = tracker.consume(
                    user_id, amount=audio_seconds
                )

        logger.debug(
            f"Consumed synthesis quota for {user_id}: "
            f"{text_length} chars, {audio_duration_seconds:.1f}s audio"
        )

        return statuses

    def check_enrollment(self, user_id: str) -> QuotaStatus:
        """Check enrollment quota."""
        return self._trackers["enrollments_day"].check(user_id)

    def consume_enrollment(self, user_id: str) -> QuotaStatus:
        """Consume enrollment quota."""
        return self._trackers["enrollments_day"].consume(user_id)

    def check_verification(self, user_id: str) -> QuotaStatus:
        """Check verification quota."""
        return self._trackers["verifications_hour"].check(user_id)

    def consume_verification(self, user_id: str) -> QuotaStatus:
        """Consume verification quota."""
        return self._trackers["verifications_hour"].consume(user_id)

    def get_all_statuses(self, user_id: str) -> Dict[str, QuotaStatus]:
        """Get status of all quotas for a user."""
        return {
            name: tracker.check(user_id)
            for name, tracker in self._trackers.items()
        }

    def is_allowed(
        self,
        user_id: str,
        operation: str,
        text_length: int = 0,
        audio_duration_seconds: float = 0.0,
    ) -> Tuple[bool, List[str]]:
        """
        Check if operation is allowed under quotas.

        Args:
            user_id: User identifier
            operation: Operation type
            text_length: Text length for synthesis
            audio_duration_seconds: Audio duration

        Returns:
            Tuple of (allowed, list of exceeded quota names)
        """
        exceeded = []

        if operation in ["synthesize", "synthesis", "stream"]:
            statuses = self.check_synthesis(
                user_id, text_length, audio_duration_seconds
            )
            exceeded = [
                name for name, status in statuses.items()
                if status.exceeded
            ]
        elif operation == "enroll":
            status = self.check_enrollment(user_id)
            if status.exceeded:
                exceeded.append("enrollments_day")
        elif operation == "verify":
            status = self.check_verification(user_id)
            if status.exceeded:
                exceeded.append("verifications_hour")

        return len(exceeded) == 0, exceeded

    def reset(
        self,
        user_id: Optional[str] = None,
        quota_name: Optional[str] = None,
    ) -> None:
        """Reset quotas."""
        with self._lock:
            if quota_name:
                tracker = self._trackers.get(quota_name)
                if tracker:
                    tracker.reset(user_id)
            else:
                for tracker in self._trackers.values():
                    tracker.reset(user_id)


# Singleton instance
_manager_instance: Optional[QuotaManager] = None


def get_quota_manager(
    config: Optional[QuotaConfig] = None,
) -> QuotaManager:
    """Get or create quota manager singleton."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = QuotaManager(config=config)
    return _manager_instance


def set_quota_manager(manager: QuotaManager) -> None:
    """Set the quota manager singleton."""
    global _manager_instance
    _manager_instance = manager
