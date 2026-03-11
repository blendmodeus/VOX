"""
Voice Boundaries
----------------

Ethical guardrails for voice cloning and voice identity protection.

Core Principles:
    1. CONSENT: Never clone a voice without verified consent
    2. IDENTITY: Protect against impersonation of public figures
    3. CONTEXT: Some voices are restricted to specific use cases
    4. HARM PREVENTION: Block voices from being used for deception

Architecture:
    Clone Request → [Consent Check] → [Identity Check] → [Context Check] → Decision

Voice Categories:
    - SYNTHETIC: Fully AI-generated voices (unrestricted)
    - CONSENTED: Cloned voices with verified consent
    - RESTRICTED: Voices with use-case limitations
    - PROTECTED: Public figures, politicians, etc. (blocked by default)
    - BLOCKED: Known bad actors, deceased individuals without estate consent

Usage:
    from axiom_vox import VoiceBoundaries, VoiceCloneRequest

    boundaries = VoiceBoundaries()

    request = VoiceCloneRequest(
        voice_id="president_biden",
        intended_use="news_parody",
        content_preview="I'm declaring free ice cream for everyone..."
    )

    decision = boundaries.check(request)
    if decision.approved:
        # Proceed with synthesis
    else:
        print(f"Blocked: {decision.reason}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class VoiceCategory(str, Enum):
    """Categories of voices with different permission levels."""
    SYNTHETIC = "synthetic"          # AI-generated, unrestricted
    CONSENTED = "consented"          # Clone with verified consent
    RESTRICTED = "restricted"        # Limited to specific use cases
    PROTECTED = "protected"          # Public figures, requires extra clearance
    BLOCKED = "blocked"              # Never allow


class UseCase(str, Enum):
    """Approved use cases for voice synthesis."""
    GENERAL = "general"              # General assistant/content
    ENTERTAINMENT = "entertainment"  # Movies, games, podcasts
    EDUCATION = "education"          # Learning materials
    ACCESSIBILITY = "accessibility"  # Assistive technology
    NEWS = "news"                    # News reading
    ADVERTISING = "advertising"      # Commercial use
    PARODY = "parody"               # Satire/parody (with disclaimers)
    PERSONAL = "personal"           # Personal use only
    IMPERSONATION = "impersonation" # Always blocked


@dataclass
class VoiceCloneRequest:
    """Request to use a cloned voice."""
    voice_id: str
    intended_use: str
    content_preview: str
    requestor_id: Optional[str] = None
    has_consent_proof: bool = False
    is_parody_labeled: bool = False
    audience: str = "public"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "intended_use": self.intended_use,
            "content_preview": self.content_preview[:200],
            "requestor_id": self.requestor_id,
            "has_consent_proof": self.has_consent_proof,
            "is_parody_labeled": self.is_parody_labeled,
            "audience": self.audience,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CloneDecision:
    """Decision from voice boundary check."""
    approved: bool
    reason: str
    voice_category: VoiceCategory
    required_disclaimers: List[str] = field(default_factory=list)
    usage_restrictions: List[str] = field(default_factory=list)
    audit_log: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "voice_category": self.voice_category.value,
            "required_disclaimers": self.required_disclaimers,
            "usage_restrictions": self.usage_restrictions,
            "audit_log": self.audit_log,
        }


class VoiceBoundaries:
    """
    Ethical guardrails for voice cloning.

    Enforces:
        - Consent verification for cloned voices
        - Identity protection for public figures
        - Use-case restrictions
        - Harm prevention
    """

    # Protected individuals - require extra clearance
    PROTECTED_PATTERNS = [
        r"\b(president|senator|congressman|governor)\b",
        r"\b(biden|trump|obama|clinton)\b",
        r"\b(pope|dalai.?lama)\b",
        r"\b(ceo|elon|musk|bezos|zuckerberg)\b",
        r"\b(celebrity|actor|actress)\b",
    ]

    # Blocked content patterns
    HARMFUL_CONTENT_PATTERNS = [
        r"\b(bomb|kill|murder|attack)\b",
        r"\b(scam|fraud|steal)\b",
        r"\b(declare.?war|military.?action)\b",
        r"\b(hate|slur|racist)\b",
    ]

    # Known synthetic voice IDs (always allowed)
    SYNTHETIC_VOICES = {
        "alloy", "echo", "fable", "onyx", "nova", "shimmer",  # OpenAI
        "en-US-Standard", "en-US-Wavenet", "en-US-Neural",     # Google
        "axiom_default", "axiom_warm", "axiom_professional",   # AXIØM
    }

    def __init__(
        self,
        require_consent_for_clones: bool = True,
        allow_parody_with_disclaimer: bool = True,
        strict_protected_enforcement: bool = True,
        db: Optional["VoxDatabase"] = None,
    ):
        self.require_consent_for_clones = require_consent_for_clones
        self.allow_parody_with_disclaimer = allow_parody_with_disclaimer
        self.strict_protected_enforcement = strict_protected_enforcement

        # Compile patterns
        self.protected_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PROTECTED_PATTERNS
        ]
        self.harmful_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.HARMFUL_CONTENT_PATTERNS
        ]

        # Persistent database (if provided)
        self.db = db

        # In-memory cache (fallback if no DB)
        self._voice_registry_cache: Dict[str, Dict[str, Any]] = {}
        self._consent_registry_cache: Set[str] = set()
        self._blocked_voices_cache: Set[str] = set()

    @property
    def voice_registry(self) -> Dict[str, Dict[str, Any]]:
        """Voice registry - from DB or cache."""
        if self.db:
            # Return DB-backed registry view
            return {v["voice_id"]: v for v in self.db.list_voices()}
        return self._voice_registry_cache

    @property
    def consent_registry(self) -> Set[str]:
        """Consent registry - from DB or cache."""
        if self.db:
            # Get all consented voice IDs from DB
            return {v["voice_id"] for v in self.db.list_voices() if v.get("consent_verified")}
        return self._consent_registry_cache

    @property
    def blocked_voices(self) -> Set[str]:
        """Blocked voices - from DB or cache."""
        if self.db:
            # Query blocked voices
            result = self.db.conn.execute("SELECT voice_id FROM blocked_voices").fetchall()
            return {row["voice_id"] for row in result}
        return self._blocked_voices_cache

    def register_voice(
        self,
        voice_id: str,
        category: VoiceCategory,
        owner_id: Optional[str] = None,
        allowed_uses: Optional[List[str]] = None,
        consent_verified: bool = False,
    ):
        """Register a voice with its permissions."""
        if self.db:
            self.db.register_voice(
                voice_id=voice_id,
                category=category.value,
                owner_id=owner_id,
                consent_verified=consent_verified,
                allowed_uses=allowed_uses,
            )
            if consent_verified:
                self.db.grant_consent(voice_id, consent_type="registration")
        else:
            self._voice_registry_cache[voice_id] = {
                "category": category,
                "owner_id": owner_id,
                "allowed_uses": allowed_uses or ["general"],
                "consent_verified": consent_verified,
                "registered_at": datetime.now().isoformat(),
            }
            if consent_verified:
                self._consent_registry_cache.add(voice_id)

    def block_voice(self, voice_id: str, reason: str):
        """Permanently block a voice."""
        if self.db:
            self.db.block_voice(voice_id, reason)
        else:
            self._blocked_voices_cache.add(voice_id)
        logger.warning(f"Voice blocked: {voice_id} - {reason}")

    def check(self, request: VoiceCloneRequest) -> CloneDecision:
        """
        Check if a voice clone request is allowed.

        Returns:
            CloneDecision with approval status and any requirements
        """
        audit_log = {
            "request": request.to_dict(),
            "checks_performed": [],
            "timestamp": datetime.now().isoformat(),
        }

        # ====================================================================
        # CHECK 1: Is this voice blocked?
        # ====================================================================
        if request.voice_id in self.blocked_voices:
            audit_log["checks_performed"].append("blocked_check: FAILED")
            return CloneDecision(
                approved=False,
                reason="Voice is permanently blocked",
                voice_category=VoiceCategory.BLOCKED,
                audit_log=audit_log,
            )
        audit_log["checks_performed"].append("blocked_check: PASSED")

        # ====================================================================
        # CHECK 2: Is this a synthetic voice? (Always allowed)
        # ====================================================================
        if self._is_synthetic_voice(request.voice_id):
            audit_log["checks_performed"].append("synthetic_check: IS_SYNTHETIC")
            return CloneDecision(
                approved=True,
                reason="Synthetic voice - no restrictions",
                voice_category=VoiceCategory.SYNTHETIC,
                audit_log=audit_log,
            )

        # ====================================================================
        # CHECK 3: Is this a protected identity?
        # ====================================================================
        is_protected, protection_reason = self._check_protected_identity(request)
        audit_log["checks_performed"].append(
            f"protected_check: {'PROTECTED' if is_protected else 'NOT_PROTECTED'}"
        )

        if is_protected:
            # Protected voices require extra clearance
            if self.strict_protected_enforcement:
                # Check if parody with disclaimer
                if request.is_parody_labeled and self.allow_parody_with_disclaimer:
                    return CloneDecision(
                        approved=True,
                        reason="Protected voice allowed for labeled parody",
                        voice_category=VoiceCategory.PROTECTED,
                        required_disclaimers=[
                            "This is a parody and does not represent the actual individual.",
                            "Voice generated by AI for entertainment purposes only.",
                        ],
                        usage_restrictions=["parody_only", "must_label"],
                        audit_log=audit_log,
                    )

                return CloneDecision(
                    approved=False,
                    reason=f"Protected identity: {protection_reason}",
                    voice_category=VoiceCategory.PROTECTED,
                    audit_log=audit_log,
                )

        # ====================================================================
        # CHECK 4: Does this voice have consent?
        # ====================================================================
        if self.require_consent_for_clones:
            has_consent = self._verify_consent(request)
            audit_log["checks_performed"].append(
                f"consent_check: {'VERIFIED' if has_consent else 'NOT_VERIFIED'}"
            )

            if not has_consent:
                return CloneDecision(
                    approved=False,
                    reason="Voice clone requires verified consent",
                    voice_category=VoiceCategory.RESTRICTED,
                    audit_log=audit_log,
                )

        # ====================================================================
        # CHECK 5: Is the content harmful?
        # ====================================================================
        is_harmful, harm_reason = self._check_harmful_content(request.content_preview)
        audit_log["checks_performed"].append(
            f"harm_check: {'HARMFUL' if is_harmful else 'SAFE'}"
        )

        if is_harmful:
            return CloneDecision(
                approved=False,
                reason=f"Harmful content detected: {harm_reason}",
                voice_category=VoiceCategory.BLOCKED,
                audit_log=audit_log,
            )

        # ====================================================================
        # CHECK 6: Is use case allowed for this voice?
        # ====================================================================
        use_allowed, use_restrictions = self._check_use_case(request)
        audit_log["checks_performed"].append(
            f"use_case_check: {'ALLOWED' if use_allowed else 'RESTRICTED'}"
        )

        if not use_allowed:
            return CloneDecision(
                approved=False,
                reason=f"Use case '{request.intended_use}' not allowed for this voice",
                voice_category=VoiceCategory.RESTRICTED,
                usage_restrictions=use_restrictions,
                audit_log=audit_log,
            )

        # ====================================================================
        # ALL CHECKS PASSED
        # ====================================================================
        return CloneDecision(
            approved=True,
            reason="All boundary checks passed",
            voice_category=VoiceCategory.CONSENTED,
            usage_restrictions=use_restrictions,
            audit_log=audit_log,
        )

    def _is_synthetic_voice(self, voice_id: str) -> bool:
        """Check if voice is a known synthetic voice."""
        # Check known synthetic prefixes
        for synthetic in self.SYNTHETIC_VOICES:
            if voice_id.lower().startswith(synthetic.lower()):
                return True

        # Check registry
        if voice_id in self.voice_registry:
            return self.voice_registry[voice_id]["category"] == VoiceCategory.SYNTHETIC

        return False

    def _check_protected_identity(
        self,
        request: VoiceCloneRequest
    ) -> tuple[bool, str]:
        """Check if voice ID suggests a protected identity."""
        voice_lower = request.voice_id.lower()

        for pattern in self.protected_patterns:
            if pattern.search(voice_lower):
                return True, f"Voice ID matches protected pattern: {pattern.pattern}"

        # Also check content for identity claims
        content_lower = request.content_preview.lower()
        identity_claims = [
            r"i am (the )?(president|senator|ceo)",
            r"as (the )?(president|pope|leader)",
            r"speaking as",
        ]
        for claim_pattern in identity_claims:
            if re.search(claim_pattern, content_lower):
                return True, f"Content contains identity claim"

        return False, ""

    def _verify_consent(self, request: VoiceCloneRequest) -> bool:
        """Verify consent for voice usage."""
        # Check if explicitly marked as having consent
        if request.has_consent_proof:
            return True

        # Check consent registry
        if request.voice_id in self.consent_registry:
            return True

        # Check voice registry
        if request.voice_id in self.voice_registry:
            return self.voice_registry[request.voice_id].get("consent_verified", False)

        return False

    def _check_harmful_content(self, content: str) -> tuple[bool, str]:
        """Check for harmful content patterns."""
        for pattern in self.harmful_patterns:
            match = pattern.search(content)
            if match:
                return True, f"Matched harmful pattern: {match.group()}"
        return False, ""

    def _check_use_case(
        self,
        request: VoiceCloneRequest
    ) -> tuple[bool, List[str]]:
        """Check if use case is allowed for this voice."""
        # Impersonation is never allowed
        if request.intended_use.lower() == "impersonation":
            return False, ["impersonation_blocked"]

        # Check voice registry for restrictions
        if request.voice_id in self.voice_registry:
            allowed_uses = self.voice_registry[request.voice_id].get("allowed_uses", [])
            if allowed_uses and request.intended_use not in allowed_uses:
                return False, [f"allowed_uses: {allowed_uses}"]

        return True, []


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_default_boundaries: Optional[VoiceBoundaries] = None


def get_boundaries() -> VoiceBoundaries:
    """Get or create the default voice boundaries."""
    global _default_boundaries
    if _default_boundaries is None:
        _default_boundaries = VoiceBoundaries()
    return _default_boundaries


def check_clone_ethics(
    voice_id: str,
    content: str,
    use_case: str = "general",
) -> tuple[bool, str]:
    """
    Quick ethics check for voice cloning.

    Returns:
        (approved, reason)
    """
    boundaries = get_boundaries()
    request = VoiceCloneRequest(
        voice_id=voice_id,
        intended_use=use_case,
        content_preview=content,
    )
    decision = boundaries.check(request)
    return decision.approved, decision.reason


# ============================================================================
# CLI DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Voice Boundaries Demo")
    print("=" * 70)

    boundaries = VoiceBoundaries()

    # Register some test voices
    boundaries.register_voice(
        "jeremy_personal",
        VoiceCategory.CONSENTED,
        owner_id="jeremy",
        consent_verified=True,
    )

    test_cases = [
        {
            "voice_id": "alloy",
            "intended_use": "general",
            "content_preview": "Hello, welcome to our podcast.",
        },
        {
            "voice_id": "president_biden_clone",
            "intended_use": "news",
            "content_preview": "I am declaring a new policy...",
        },
        {
            "voice_id": "random_clone",
            "intended_use": "general",
            "content_preview": "This is a test message.",
        },
        {
            "voice_id": "jeremy_personal",
            "intended_use": "personal",
            "content_preview": "Hey, just leaving a note for myself.",
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Voice: {tc['voice_id']}")
        print(f"Use: {tc['intended_use']}")

        request = VoiceCloneRequest(**tc)
        decision = boundaries.check(request)

        status = "✓ APPROVED" if decision.approved else "✗ BLOCKED"
        print(f"Decision: {status}")
        print(f"Reason: {decision.reason}")
        print(f"Category: {decision.voice_category.value}")

        if decision.required_disclaimers:
            print(f"Disclaimers: {decision.required_disclaimers}")
