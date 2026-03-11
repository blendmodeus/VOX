"""
Unified Consent Registry
------------------------

Central consent management for all VØX voice operations.

Unifies consent across:
    - Voice synthesis
    - Voice cloning
    - Biometric enrollment
    - Streaming
    - Commercial use

AXIØM Phase 6: System - "Integrate the parts"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from enum import Enum

from .models import ConsentScope, ConsentResult

logger = logging.getLogger(__name__)


@dataclass
class ConsentRecord:
    """A single consent record."""
    consent_id: int
    voice_id: str
    owner_id: str
    scopes: Set[ConsentScope]
    granted_at: float
    expires_at: Optional[float] = None
    revoked: bool = False
    revoked_at: Optional[float] = None
    proof_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsentQuery:
    """Query for consent check."""
    voice_id: str
    required_scopes: List[ConsentScope]
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class UnifiedConsentRegistry:
    """
    Central registry for all voice-related consent.

    Provides unified interface to check consent across:
        - Voice cloning consent (VoiceBoundaries)
        - Biometric consent (BiometricConsentManager)
        - Synthesis consent (VoxDatabase)
    """

    def __init__(self, db=None):
        """
        Initialize unified consent registry.

        Args:
            db: VoxDatabase instance
        """
        self._db = db
        self._consent_cache: Dict[str, ConsentRecord] = {}
        self._cache_ttl = 300  # 5 minutes

    @property
    def db(self):
        """Get database, lazy loading if needed."""
        if self._db is None:
            from ..persistence import get_database
            self._db = get_database()
        return self._db

    def check_consent(
        self,
        voice_id: str,
        required_scopes: List[ConsentScope],
        user_id: Optional[str] = None,
    ) -> ConsentResult:
        """
        Check if consent exists for required scopes.

        Args:
            voice_id: Voice ID to check
            required_scopes: List of required consent scopes
            user_id: Optional user ID for ownership check

        Returns:
            ConsentResult
        """
        # Check cache first
        cache_key = f"{voice_id}:{':'.join(s.value for s in required_scopes)}"
        cached = self._get_cached_consent(cache_key)
        if cached:
            return cached

        # Aggregate consent from all sources
        granted_scopes: Set[ConsentScope] = set()
        restrictions: List[str] = []
        consent_id = None
        expires_at = None

        # Check voice boundaries consent (for cloning)
        if ConsentScope.CLONING in required_scopes:
            clone_consent = self._check_clone_consent(voice_id)
            if clone_consent:
                granted_scopes.add(ConsentScope.CLONING)
                if clone_consent.get("restrictions"):
                    restrictions.extend(clone_consent["restrictions"])

        # Check biometric consent
        if ConsentScope.BIOMETRIC in required_scopes:
            bio_consent = self._check_biometric_consent(voice_id)
            if bio_consent:
                granted_scopes.add(ConsentScope.BIOMETRIC)
                consent_id = bio_consent.get("consent_id")
                expires_at = bio_consent.get("expires_at")

        # Check synthesis consent (general usage)
        if ConsentScope.SYNTHESIS in required_scopes:
            synth_consent = self._check_synthesis_consent(voice_id, user_id)
            if synth_consent:
                granted_scopes.add(ConsentScope.SYNTHESIS)
                # Synthesis consent may include streaming
                if synth_consent.get("streaming_allowed"):
                    granted_scopes.add(ConsentScope.STREAMING)

        # Check commercial consent
        if ConsentScope.COMMERCIAL in required_scopes:
            commercial_consent = self._check_commercial_consent(voice_id)
            if commercial_consent:
                granted_scopes.add(ConsentScope.COMMERCIAL)
            else:
                restrictions.append("Non-commercial use only")

        # Check third-party consent
        if ConsentScope.THIRD_PARTY in required_scopes:
            tp_consent = self._check_third_party_consent(voice_id)
            if tp_consent:
                granted_scopes.add(ConsentScope.THIRD_PARTY)
            else:
                restrictions.append("First-party use only")

        # Determine if all required scopes are granted
        missing_scopes = set(required_scopes) - granted_scopes
        granted = len(missing_scopes) == 0

        result = ConsentResult(
            granted=granted,
            scopes=list(granted_scopes),
            consent_id=consent_id,
            expires_at=expires_at,
            restrictions=restrictions,
            message=self._build_consent_message(granted, missing_scopes),
        )

        # Cache result
        self._cache_consent(cache_key, result)

        return result

    def grant_consent(
        self,
        voice_id: str,
        scopes: List[ConsentScope],
        owner_id: str,
        proof: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Grant consent for specified scopes.

        Args:
            voice_id: Voice ID
            scopes: Consent scopes to grant
            owner_id: Owner granting consent
            proof: Proof of consent
            expires_in_days: Expiration in days
            metadata: Additional metadata

        Returns:
            Consent record ID
        """
        now = time.time()
        expires_at = None
        if expires_in_days:
            expires_at = now + (expires_in_days * 86400)

        # Store unified consent
        for scope in scopes:
            consent_type = f"unified_{scope.value}"

            self.db.grant_consent(
                voice_id=voice_id,
                consent_type=consent_type,
                proof=proof,
                granted_by=owner_id,
                expires_in_days=expires_in_days,
                metadata={
                    **(metadata or {}),
                    "scope": scope.value,
                    "unified_registry": True,
                },
            )

        # Invalidate cache
        self._invalidate_cache(voice_id)

        logger.info(
            f"Granted consent: voice={voice_id}, scopes={[s.value for s in scopes]}"
        )

        return 1  # Return success indicator

    def revoke_consent(
        self,
        voice_id: str,
        scopes: Optional[List[ConsentScope]] = None,
        reason: str = "",
    ) -> bool:
        """
        Revoke consent for specified scopes.

        Args:
            voice_id: Voice ID
            scopes: Scopes to revoke (None = all)
            reason: Reason for revocation

        Returns:
            True if any consent was revoked
        """
        scopes_to_revoke = scopes or list(ConsentScope)
        revoked = False

        for scope in scopes_to_revoke:
            consent_type = f"unified_{scope.value}"

            if self.db.revoke_consent(voice_id, consent_type=consent_type):
                revoked = True
                logger.info(f"Revoked consent: voice={voice_id}, scope={scope.value}")

        # Also revoke from component systems
        if ConsentScope.BIOMETRIC in scopes_to_revoke:
            self._revoke_biometric_consent(voice_id)

        if ConsentScope.CLONING in scopes_to_revoke:
            self._revoke_clone_consent(voice_id)

        # Invalidate cache
        self._invalidate_cache(voice_id)

        return revoked

    def get_consent_status(self, voice_id: str) -> Dict[str, Any]:
        """
        Get complete consent status for a voice.

        Args:
            voice_id: Voice ID

        Returns:
            Complete consent status
        """
        status = {
            "voice_id": voice_id,
            "scopes": {},
            "has_any_consent": False,
            "restrictions": [],
        }

        for scope in ConsentScope:
            result = self.check_consent(voice_id, [scope])
            status["scopes"][scope.value] = {
                "granted": result.granted,
                "expires_at": result.expires_at,
            }
            if result.granted:
                status["has_any_consent"] = True
            status["restrictions"].extend(result.restrictions)

        # Deduplicate restrictions
        status["restrictions"] = list(set(status["restrictions"]))

        return status

    def _check_clone_consent(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Check voice cloning consent."""
        try:
            from ..voice_boundaries import VoiceBoundaries, VoiceCategory

            boundaries = VoiceBoundaries()
            voice_info = self.db.get_voice(voice_id)

            if not voice_info:
                return None

            category = voice_info.get("category", "")
            if category == VoiceCategory.CONSENTED.value:
                return {
                    "granted": True,
                    "restrictions": voice_info.get("allowed_uses", []),
                }
            elif category == VoiceCategory.SYNTHETIC.value:
                return {"granted": True, "restrictions": []}

            return None

        except Exception as e:
            logger.debug(f"Clone consent check error: {e}")
            return None

    def _check_biometric_consent(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Check biometric consent."""
        try:
            from ..biometrics import get_consent_manager, ConsentType

            manager = get_consent_manager()
            result = manager.check_consent(voice_id, ConsentType.ENROLLMENT)

            if result.status.value == "valid":
                return {
                    "granted": True,
                    "consent_id": result.consent_id,
                    "expires_at": result.expires_at,
                }

            return None

        except Exception as e:
            logger.debug(f"Biometric consent check error: {e}")
            return None

    def _check_synthesis_consent(
        self,
        voice_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Check synthesis consent."""
        try:
            # Check unified consent first
            consent = self.db.get_active_consent(
                voice_id,
                consent_type="unified_synthesis",
            )

            if consent:
                return {
                    "granted": True,
                    "streaming_allowed": consent.get("metadata", {}).get(
                        "streaming_allowed", True
                    ),
                }

            # Check voice registration
            voice_info = self.db.get_voice(voice_id)
            if voice_info:
                # Synthetic voices always have synthesis consent
                if voice_info.get("category") == "synthetic":
                    return {"granted": True, "streaming_allowed": True}

                # Consented voices have synthesis consent
                if voice_info.get("consent_verified"):
                    allowed_uses = voice_info.get("allowed_uses", [])
                    if "general" in allowed_uses or "synthesis" in allowed_uses:
                        return {"granted": True, "streaming_allowed": True}

            return None

        except Exception as e:
            logger.debug(f"Synthesis consent check error: {e}")
            return None

    def _check_commercial_consent(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Check commercial use consent."""
        try:
            consent = self.db.get_active_consent(
                voice_id,
                consent_type="unified_commercial",
            )

            if consent:
                return {"granted": True}

            # Check voice metadata
            voice_info = self.db.get_voice(voice_id)
            if voice_info:
                allowed_uses = voice_info.get("allowed_uses", [])
                if "commercial" in allowed_uses or "advertising" in allowed_uses:
                    return {"granted": True}

            return None

        except Exception as e:
            logger.debug(f"Commercial consent check error: {e}")
            return None

    def _check_third_party_consent(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Check third-party sharing consent."""
        try:
            consent = self.db.get_active_consent(
                voice_id,
                consent_type="unified_third_party",
            )

            if consent:
                return {"granted": True}

            return None

        except Exception as e:
            logger.debug(f"Third-party consent check error: {e}")
            return None

    def _revoke_biometric_consent(self, voice_id: str) -> bool:
        """Revoke biometric consent in component system."""
        try:
            from ..biometrics import get_consent_manager

            manager = get_consent_manager()
            return manager.revoke_consent(voice_id)
        except Exception as e:
            logger.debug(f"Biometric consent revocation error: {e}")
            return False

    def _revoke_clone_consent(self, voice_id: str) -> bool:
        """Revoke clone consent in component system."""
        try:
            # Update voice category to blocked
            voice_info = self.db.get_voice(voice_id)
            if voice_info:
                self.db.register_voice(
                    voice_id=voice_id,
                    category="blocked",
                    consent_verified=False,
                )
                return True
            return False
        except Exception as e:
            logger.debug(f"Clone consent revocation error: {e}")
            return False

    def _build_consent_message(
        self,
        granted: bool,
        missing_scopes: Set[ConsentScope],
    ) -> str:
        """Build human-readable consent message."""
        if granted:
            return "All required consent granted"

        missing = [s.value for s in missing_scopes]
        return f"Missing consent for: {', '.join(missing)}"

    def _get_cached_consent(self, key: str) -> Optional[ConsentResult]:
        """Get consent from cache if valid."""
        if key not in self._consent_cache:
            return None

        record = self._consent_cache[key]
        # Check if expired
        if hasattr(record, '_cached_at'):
            if time.time() - record._cached_at > self._cache_ttl:
                del self._consent_cache[key]
                return None

        return record

    def _cache_consent(self, key: str, result: ConsentResult) -> None:
        """Cache consent result."""
        result._cached_at = time.time()
        self._consent_cache[key] = result

    def _invalidate_cache(self, voice_id: str) -> None:
        """Invalidate all cache entries for a voice."""
        keys_to_remove = [k for k in self._consent_cache if k.startswith(f"{voice_id}:")]
        for key in keys_to_remove:
            del self._consent_cache[key]


# Singleton instance
_registry_instance: Optional[UnifiedConsentRegistry] = None


def get_consent_registry(db=None) -> UnifiedConsentRegistry:
    """Get or create consent registry singleton."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = UnifiedConsentRegistry(db=db)
    return _registry_instance


def set_consent_registry(registry: UnifiedConsentRegistry) -> None:
    """Set the consent registry singleton."""
    global _registry_instance
    _registry_instance = registry
