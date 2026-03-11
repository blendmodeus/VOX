"""
Biometric Consent Manager
-------------------------

Consent verification for voice biometric enrollment and operations.

Ensures:
    - Explicit consent before biometric enrollment
    - Consent token validation
    - Audit trail for consent decisions
    - Revocation support

AXIØM Phase 5: Resonance - "finding signature frequency"
"""

import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ConsentType(str, Enum):
    """Types of biometric consent."""
    ENROLLMENT = "enrollment"  # Initial biometric enrollment
    VERIFICATION = "verification"  # Ongoing verification use
    UPDATE = "update"  # Template updates
    DATA_RETENTION = "data_retention"  # Storing biometric data
    THIRD_PARTY = "third_party"  # Sharing with third parties


class ConsentStatus(str, Enum):
    """Status of consent check."""
    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    NOT_FOUND = "not_found"
    INVALID = "invalid"
    MISSING = "missing"


@dataclass
class ConsentToken:
    """Consent token for biometric operations."""
    token_id: str
    voice_id: str
    owner_id: str
    consent_types: List[ConsentType]
    granted_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    signature: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsentCheckResult:
    """Result of consent verification."""
    status: ConsentStatus
    consent_id: Optional[int] = None
    consent_types: List[ConsentType] = field(default_factory=list)
    message: str = ""
    expires_at: Optional[float] = None


class BiometricConsentManager:
    """
    Manage consent for biometric operations.

    Integrates with VoxDatabase consent registry.
    """

    # Default token validity: 1 year
    DEFAULT_EXPIRY_DAYS = 365

    def __init__(self, db=None, secret_key: Optional[str] = None):
        """
        Initialize consent manager.

        Args:
            db: VoxDatabase instance
            secret_key: Secret for signing tokens (auto-generated if not provided)
        """
        self.db = db
        self._secret_key = secret_key or secrets.token_hex(32)

    def _get_db(self):
        """Get database, lazy loading if needed."""
        if self.db is None:
            from ..persistence import get_database
            self.db = get_database()
        return self.db

    def generate_token(
        self,
        voice_id: str,
        owner_id: str,
        consent_types: Optional[List[ConsentType]] = None,
        expiry_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConsentToken:
        """
        Generate a consent token for biometric operations.

        Args:
            voice_id: Voice ID the consent applies to
            owner_id: Owner granting consent
            consent_types: Types of consent being granted
            expiry_days: Days until expiration
            metadata: Additional consent metadata

        Returns:
            ConsentToken
        """
        token_id = secrets.token_urlsafe(24)
        granted_at = time.time()

        expiry_days = expiry_days or self.DEFAULT_EXPIRY_DAYS
        expires_at = granted_at + (expiry_days * 86400)

        consent_types = consent_types or [
            ConsentType.ENROLLMENT,
            ConsentType.VERIFICATION,
            ConsentType.UPDATE,
            ConsentType.DATA_RETENTION,
        ]

        # Generate signature
        data_to_sign = f"{token_id}:{voice_id}:{owner_id}:{expires_at}"
        signature = hmac.new(
            self._secret_key.encode(),
            data_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()

        return ConsentToken(
            token_id=token_id,
            voice_id=voice_id,
            owner_id=owner_id,
            consent_types=consent_types,
            granted_at=granted_at,
            expires_at=expires_at,
            signature=signature,
            metadata=metadata or {},
        )

    def verify_token(self, token: ConsentToken) -> ConsentCheckResult:
        """
        Verify a consent token.

        Args:
            token: ConsentToken to verify

        Returns:
            ConsentCheckResult
        """
        # Verify signature
        data_to_sign = f"{token.token_id}:{token.voice_id}:{token.owner_id}:{token.expires_at}"
        expected_signature = hmac.new(
            self._secret_key.encode(),
            data_to_sign.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(token.signature, expected_signature):
            return ConsentCheckResult(
                status=ConsentStatus.INVALID,
                message="Invalid token signature",
            )

        # Check expiration
        if token.expires_at and time.time() > token.expires_at:
            return ConsentCheckResult(
                status=ConsentStatus.EXPIRED,
                message="Consent token has expired",
                expires_at=token.expires_at,
            )

        return ConsentCheckResult(
            status=ConsentStatus.VALID,
            consent_types=token.consent_types,
            expires_at=token.expires_at,
            message="Consent verified",
        )

    def verify_token_string(
        self,
        token_string: str,
        voice_id: str,
        required_type: ConsentType = ConsentType.ENROLLMENT,
    ) -> ConsentCheckResult:
        """
        Verify a consent token string.

        Token format: base64(token_id:voice_id:owner_id:expires_at:signature)

        Args:
            token_string: Encoded consent token
            voice_id: Expected voice ID
            required_type: Required consent type

        Returns:
            ConsentCheckResult
        """
        import base64

        try:
            decoded = base64.urlsafe_b64decode(token_string.encode()).decode()
            parts = decoded.split(":")

            if len(parts) != 5:
                return ConsentCheckResult(
                    status=ConsentStatus.INVALID,
                    message="Malformed consent token",
                )

            token_id, token_voice_id, owner_id, expires_at_str, signature = parts

            if token_voice_id != voice_id:
                return ConsentCheckResult(
                    status=ConsentStatus.INVALID,
                    message="Token voice_id mismatch",
                )

            expires_at = float(expires_at_str)

            # Verify signature
            data_to_sign = f"{token_id}:{token_voice_id}:{owner_id}:{expires_at}"
            expected_signature = hmac.new(
                self._secret_key.encode(),
                data_to_sign.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                return ConsentCheckResult(
                    status=ConsentStatus.INVALID,
                    message="Invalid token signature",
                )

            # Check expiration
            if time.time() > expires_at:
                return ConsentCheckResult(
                    status=ConsentStatus.EXPIRED,
                    message="Consent token has expired",
                    expires_at=expires_at,
                )

            return ConsentCheckResult(
                status=ConsentStatus.VALID,
                consent_types=[required_type],  # Assume granted for required type
                expires_at=expires_at,
                message="Consent verified",
            )

        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return ConsentCheckResult(
                status=ConsentStatus.INVALID,
                message=f"Token verification error: {str(e)}",
            )

    def encode_token(self, token: ConsentToken) -> str:
        """
        Encode a consent token to a string.

        Args:
            token: ConsentToken to encode

        Returns:
            Encoded token string
        """
        import base64

        data = f"{token.token_id}:{token.voice_id}:{token.owner_id}:{token.expires_at}:{token.signature}"
        return base64.urlsafe_b64encode(data.encode()).decode()

    def record_consent(
        self,
        voice_id: str,
        consent_type: ConsentType,
        owner_id: str,
        proof: Optional[str] = None,
        expiry_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record consent in the database.

        Args:
            voice_id: Voice ID
            consent_type: Type of consent
            owner_id: Owner granting consent
            proof: Proof of consent (signature, hash, etc.)
            expiry_days: Days until expiration
            metadata: Additional metadata

        Returns:
            Consent record ID
        """
        db = self._get_db()

        consent_id = db.grant_consent(
            voice_id=voice_id,
            consent_type=f"biometric_{consent_type.value}",
            proof=proof,
            granted_by=owner_id,
            expires_in_days=expiry_days or self.DEFAULT_EXPIRY_DAYS,
            metadata=metadata,
        )

        logger.info(
            f"Recorded biometric consent: voice={voice_id}, "
            f"type={consent_type.value}, owner={owner_id}"
        )

        return consent_id

    def check_consent(
        self,
        voice_id: str,
        consent_type: ConsentType = ConsentType.ENROLLMENT,
    ) -> ConsentCheckResult:
        """
        Check if consent exists for a voice and operation type.

        Args:
            voice_id: Voice ID
            consent_type: Type of consent to check

        Returns:
            ConsentCheckResult
        """
        db = self._get_db()

        consent = db.get_active_consent(
            voice_id,
            consent_type=f"biometric_{consent_type.value}"
        )

        if not consent:
            return ConsentCheckResult(
                status=ConsentStatus.NOT_FOUND,
                message=f"No {consent_type.value} consent found for voice",
            )

        # Check if consent is still valid
        if consent.get("revoked"):
            return ConsentCheckResult(
                status=ConsentStatus.REVOKED,
                consent_id=consent.get("id"),
                message="Consent has been revoked",
            )

        expires_at = consent.get("expires_at")
        if expires_at and time.time() > expires_at:
            return ConsentCheckResult(
                status=ConsentStatus.EXPIRED,
                consent_id=consent.get("id"),
                expires_at=expires_at,
                message="Consent has expired",
            )

        return ConsentCheckResult(
            status=ConsentStatus.VALID,
            consent_id=consent.get("id"),
            consent_types=[consent_type],
            expires_at=expires_at,
            message="Consent verified",
        )

    def revoke_consent(
        self,
        voice_id: str,
        consent_type: Optional[ConsentType] = None,
        reason: str = "",
    ) -> bool:
        """
        Revoke consent for a voice.

        Args:
            voice_id: Voice ID
            consent_type: Specific type to revoke (None = all)
            reason: Reason for revocation

        Returns:
            True if consent was revoked
        """
        db = self._get_db()

        if consent_type:
            revoked = db.revoke_consent(
                voice_id,
                consent_type=f"biometric_{consent_type.value}",
            )
        else:
            # Revoke all biometric consents
            revoked = False
            for ct in ConsentType:
                if db.revoke_consent(
                    voice_id,
                    consent_type=f"biometric_{ct.value}",
                ):
                    revoked = True

        if revoked:
            logger.info(
                f"Revoked biometric consent: voice={voice_id}, "
                f"type={consent_type.value if consent_type else 'all'}"
            )

        return revoked

    def require_consent(
        self,
        voice_id: str,
        consent_token: Optional[str] = None,
        consent_type: ConsentType = ConsentType.ENROLLMENT,
    ) -> ConsentCheckResult:
        """
        Require valid consent for an operation.

        Checks token first, then falls back to database.

        Args:
            voice_id: Voice ID
            consent_token: Optional consent token string
            consent_type: Required consent type

        Returns:
            ConsentCheckResult
        """
        # First try token verification
        if consent_token:
            result = self.verify_token_string(
                consent_token,
                voice_id,
                required_type=consent_type,
            )
            if result.status == ConsentStatus.VALID:
                return result

        # Fall back to database check
        return self.check_consent(voice_id, consent_type)


# Singleton instance
_consent_manager: Optional[BiometricConsentManager] = None


def get_consent_manager(db=None) -> BiometricConsentManager:
    """Get or create consent manager singleton."""
    global _consent_manager
    if _consent_manager is None:
        _consent_manager = BiometricConsentManager(db=db)
    return _consent_manager


def set_consent_manager(manager: BiometricConsentManager) -> None:
    """Set the consent manager singleton."""
    global _consent_manager
    _consent_manager = manager
