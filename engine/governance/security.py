"""
Security Manager
----------------

Encryption, access control, and audit logging for voice operations.

AXIØM Phase 7: Constrain - "What limits clarify the solution?"
"""

import logging
import time
import threading
import hashlib
import hmac
import secrets
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Set, Any, Callable
import base64
import json

from .models import (
    SecurityConfig,
    SecurityLevel,
    AccessToken,
    AuditEntry,
)

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Authentication failed."""
    pass


class AuthorizationError(Exception):
    """Authorization failed."""
    pass


class AccessController:
    """
    Role-based access control for voice operations.

    Manages:
        - API key validation
        - Token-based authentication
        - Permission checking
        - Scope verification
    """

    # Default roles and permissions
    DEFAULT_ROLES = {
        "admin": {
            "synthesize", "enroll", "verify", "delete", "manage",
            "read_audit", "manage_users", "manage_policies",
        },
        "user": {
            "synthesize", "verify",
        },
        "developer": {
            "synthesize", "enroll", "verify", "read_audit",
        },
        "viewer": {
            "synthesize",
        },
    }

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
    ):
        """
        Initialize access controller.

        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._tokens: Dict[str, AccessToken] = {}
        self._user_roles: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()

        # Key derivation secret (should be loaded from secure config)
        self._secret = secrets.token_bytes(32)

    def register_api_key(
        self,
        api_key: str,
        user_id: str,
        scopes: Optional[Set[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> None:
        """
        Register an API key.

        Args:
            api_key: The API key
            user_id: Associated user ID
            scopes: Allowed scopes
            expires_in_days: Optional expiration
        """
        with self._lock:
            key_hash = self._hash_key(api_key)
            expires_at = None
            if expires_in_days:
                expires_at = time.time() + (expires_in_days * 86400)

            self._api_keys[key_hash] = {
                "user_id": user_id,
                "scopes": scopes or {"synthesize"},
                "created_at": time.time(),
                "expires_at": expires_at,
            }

            logger.info(f"Registered API key for user {user_id}")

    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Validate an API key.

        Args:
            api_key: The API key to validate

        Returns:
            Key metadata

        Raises:
            AuthenticationError: If key is invalid
        """
        key_hash = self._hash_key(api_key)

        with self._lock:
            key_data = self._api_keys.get(key_hash)

            if not key_data:
                raise AuthenticationError("Invalid API key")

            # Check expiration
            if key_data.get("expires_at"):
                if time.time() > key_data["expires_at"]:
                    raise AuthenticationError("API key expired")

            return key_data

    def create_token(
        self,
        user_id: str,
        scopes: Set[str],
        expires_in_seconds: int = 3600,
    ) -> AccessToken:
        """
        Create an access token.

        Args:
            user_id: User identifier
            scopes: Token scopes
            expires_in_seconds: Token lifetime

        Returns:
            AccessToken
        """
        token_id = f"tok_{uuid.uuid4().hex}"
        now = time.time()

        token = AccessToken(
            token_id=token_id,
            user_id=user_id,
            scopes=scopes,
            issued_at=now,
            expires_at=now + expires_in_seconds,
        )

        with self._lock:
            self._tokens[token_id] = token

        logger.debug(f"Created token {token_id} for user {user_id}")
        return token

    def validate_token(self, token_id: str) -> AccessToken:
        """
        Validate an access token.

        Args:
            token_id: Token identifier

        Returns:
            AccessToken

        Raises:
            AuthenticationError: If token is invalid
        """
        with self._lock:
            token = self._tokens.get(token_id)

            if not token:
                raise AuthenticationError("Invalid token")

            if token.revoked:
                raise AuthenticationError("Token revoked")

            if time.time() > token.expires_at:
                raise AuthenticationError("Token expired")

            return token

    def revoke_token(self, token_id: str) -> bool:
        """Revoke an access token."""
        with self._lock:
            token = self._tokens.get(token_id)
            if token:
                token.revoked = True
                return True
            return False

    def check_permission(
        self,
        user_id: str,
        permission: str,
        resource_id: Optional[str] = None,
    ) -> bool:
        """
        Check if user has permission.

        Args:
            user_id: User identifier
            permission: Permission to check
            resource_id: Optional resource identifier

        Returns:
            True if permitted
        """
        with self._lock:
            roles = self._user_roles.get(user_id, {"user"})

            for role in roles:
                role_perms = self.DEFAULT_ROLES.get(role, set())
                if permission in role_perms:
                    return True

            return False

    def require_permission(
        self,
        user_id: str,
        permission: str,
        resource_id: Optional[str] = None,
    ) -> None:
        """
        Require a permission (raise if not present).

        Args:
            user_id: User identifier
            permission: Required permission
            resource_id: Optional resource identifier

        Raises:
            AuthorizationError: If permission not granted
        """
        if not self.check_permission(user_id, permission, resource_id):
            raise AuthorizationError(
                f"Permission denied: {permission} for user {user_id}"
            )

    def set_user_roles(self, user_id: str, roles: Set[str]) -> None:
        """Set roles for a user."""
        with self._lock:
            self._user_roles[user_id] = roles

    def get_user_roles(self, user_id: str) -> Set[str]:
        """Get roles for a user."""
        return self._user_roles.get(user_id, {"user"})

    def _hash_key(self, key: str) -> str:
        """Hash an API key for storage."""
        return hmac.new(
            self._secret,
            key.encode(),
            hashlib.sha256,
        ).hexdigest()

    def cleanup_expired(self) -> int:
        """Clean up expired tokens."""
        now = time.time()
        cleaned = 0

        with self._lock:
            expired = [
                tid for tid, token in self._tokens.items()
                if token.expires_at < now
            ]
            for tid in expired:
                del self._tokens[tid]
                cleaned += 1

        return cleaned


class AuditLogger:
    """
    Audit logging for voice operations.

    Records all significant operations for compliance
    and security monitoring.
    """

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        max_entries: int = 10000,
        db=None,
    ):
        """
        Initialize audit logger.

        Args:
            config: Security configuration
            max_entries: Maximum in-memory entries
            db: Optional database for persistence
        """
        self.config = config or SecurityConfig()
        self._entries: deque = deque(maxlen=max_entries)
        self._lock = threading.RLock()
        self._db = db
        self._callbacks: List[Callable[[AuditEntry], None]] = []

    def log(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        result: str,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log an audit event.

        Args:
            user_id: User who performed action
            action: Action performed
            resource_type: Type of resource
            resource_id: Resource identifier
            result: Result (success, failure, blocked)
            ip_address: Client IP address
            details: Additional details

        Returns:
            AuditEntry
        """
        entry = AuditEntry(
            entry_id=f"audit_{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            result=result,
            ip_address=ip_address,
            details=details or {},
        )

        with self._lock:
            self._entries.append(entry)

        # Persist if database available
        if self._db:
            self._persist_entry(entry)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"Audit callback error: {e}")

        logger.debug(
            f"Audit: {user_id} {action} {resource_type}/{resource_id} -> {result}"
        )

        return entry

    def log_synthesis(
        self,
        user_id: str,
        voice_id: str,
        text_length: int,
        result: str,
        **kwargs,
    ) -> AuditEntry:
        """Log a synthesis operation."""
        return self.log(
            user_id=user_id,
            action="synthesize",
            resource_type="voice",
            resource_id=voice_id,
            result=result,
            details={"text_length": text_length, **kwargs},
        )

    def log_enrollment(
        self,
        user_id: str,
        voice_id: str,
        result: str,
        **kwargs,
    ) -> AuditEntry:
        """Log an enrollment operation."""
        return self.log(
            user_id=user_id,
            action="enroll",
            resource_type="biometric",
            resource_id=voice_id,
            result=result,
            details=kwargs,
        )

    def log_verification(
        self,
        user_id: str,
        voice_id: str,
        verified: bool,
        similarity: float,
        **kwargs,
    ) -> AuditEntry:
        """Log a verification operation."""
        return self.log(
            user_id=user_id,
            action="verify",
            resource_type="biometric",
            resource_id=voice_id,
            result="success" if verified else "failure",
            details={"verified": verified, "similarity": similarity, **kwargs},
        )

    def log_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        granted: bool,
        **kwargs,
    ) -> AuditEntry:
        """Log an access control decision."""
        return self.log(
            user_id=user_id,
            action="access",
            resource_type=resource_type,
            resource_id=resource_id,
            result="granted" if granted else "denied",
            details=kwargs,
        )

    def get_entries(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Get audit entries with filters.

        Args:
            user_id: Filter by user
            action: Filter by action
            resource_type: Filter by resource type
            since: Filter by timestamp
            limit: Maximum entries

        Returns:
            List of matching entries
        """
        with self._lock:
            entries = list(self._entries)

        # Apply filters
        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        if action:
            entries = [e for e in entries if e.action == action]
        if resource_type:
            entries = [e for e in entries if e.resource_type == resource_type]
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        # Sort by timestamp (newest first) and limit
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def add_callback(self, callback: Callable[[AuditEntry], None]) -> None:
        """Add a callback for new audit entries."""
        self._callbacks.append(callback)

    def _persist_entry(self, entry: AuditEntry) -> None:
        """Persist entry to database."""
        try:
            # Would use self._db to persist
            pass
        except Exception as e:
            logger.error(f"Failed to persist audit entry: {e}")


class SecurityManager:
    """
    Central security management for voice operations.

    Coordinates:
        - Access control
        - Audit logging
        - Encryption (placeholder for actual implementation)
    """

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        db=None,
    ):
        """
        Initialize security manager.

        Args:
            config: Security configuration
            db: Optional database
        """
        self.config = config or SecurityConfig()
        self.access = AccessController(config)
        self.audit = AuditLogger(config, db=db)

    def authenticate(
        self,
        api_key: Optional[str] = None,
        token_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Authenticate a request.

        Args:
            api_key: API key
            token_id: Access token

        Returns:
            Authentication context

        Raises:
            AuthenticationError: If authentication fails
        """
        if api_key:
            key_data = self.access.validate_api_key(api_key)
            return {
                "user_id": key_data["user_id"],
                "scopes": key_data["scopes"],
                "auth_type": "api_key",
            }

        if token_id:
            token = self.access.validate_token(token_id)
            return {
                "user_id": token.user_id,
                "scopes": token.scopes,
                "auth_type": "token",
            }

        if not self.config.require_api_key:
            return {
                "user_id": "anonymous",
                "scopes": {"synthesize"},
                "auth_type": "anonymous",
            }

        raise AuthenticationError("Authentication required")

    def authorize(
        self,
        user_id: str,
        operation: str,
        resource_id: Optional[str] = None,
        scopes: Optional[Set[str]] = None,
    ) -> bool:
        """
        Authorize an operation.

        Args:
            user_id: User identifier
            operation: Operation to authorize
            resource_id: Optional resource
            scopes: User's scopes

        Returns:
            True if authorized
        """
        # Check scope
        if scopes and operation not in scopes:
            self.audit.log_access(
                user_id=user_id,
                resource_type="operation",
                resource_id=operation,
                granted=False,
                reason="scope_missing",
            )
            return False

        # Check permission
        if not self.access.check_permission(user_id, operation, resource_id):
            self.audit.log_access(
                user_id=user_id,
                resource_type="operation",
                resource_id=operation,
                granted=False,
                reason="permission_denied",
            )
            return False

        return True

    def generate_api_key(self, prefix: str = "vox") -> str:
        """Generate a new API key."""
        key_bytes = secrets.token_bytes(24)
        key = f"{prefix}_{base64.urlsafe_b64encode(key_bytes).decode().rstrip('=')}"
        return key


# Singleton instance
_manager_instance: Optional[SecurityManager] = None


def get_security_manager(
    config: Optional[SecurityConfig] = None,
) -> SecurityManager:
    """Get or create security manager singleton."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SecurityManager(config=config)
    return _manager_instance


def set_security_manager(manager: SecurityManager) -> None:
    """Set the security manager singleton."""
    global _manager_instance
    _manager_instance = manager
