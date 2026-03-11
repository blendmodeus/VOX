"""
PRIME Voice Identity Manager
-----------------------------

Manages PRIME's locked voice identity - the immutable vocal fingerprint
that ensures PRIME always sounds like PRIME.

The identity system provides:
- Locked 8-dimensional voice vector (no drift allowed)
- Biometric enrollment and verification anchor
- Voice DNA target enforcement
- Identity consistency monitoring
- Tamper detection (voice spoofing prevention)
"""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    IdentityLockLevel,
    PrimeVocalDNA,
    PrimeVoiceIdentity,
    PrimeVoiceVector,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Identity Configuration
# =============================================================================

@dataclass
class IdentityConfig:
    """Configuration for PRIME voice identity management."""
    # Lock enforcement
    lock_level: IdentityLockLevel = IdentityLockLevel.LOCKED
    allow_calibration: bool = False

    # Biometric verification
    enable_biometric_anchor: bool = True
    similarity_threshold: float = 0.75
    verification_interval: int = 10  # Verify every N utterances

    # Drift detection
    max_vector_drift: float = 0.05   # Max allowed deviation per dimension
    drift_check_window: int = 20     # Check drift over last N utterances

    # Integrity
    enable_integrity_hash: bool = True
    tamper_alert_threshold: int = 3  # Alert after N failed checks

    # Persistence
    identity_file: Optional[str] = None  # Path to save/load identity


# =============================================================================
# Identity Integrity
# =============================================================================

@dataclass
class IdentityCheckResult:
    """Result of an identity verification check."""
    passed: bool = True
    similarity_score: float = 1.0
    drift_detected: bool = False
    drift_dimensions: List[str] = field(default_factory=list)
    tamper_suspected: bool = False
    integrity_hash_valid: bool = True
    message: str = "Identity verified"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "similarity_score": self.similarity_score,
            "drift_detected": self.drift_detected,
            "drift_dimensions": self.drift_dimensions,
            "tamper_suspected": self.tamper_suspected,
            "integrity_hash_valid": self.integrity_hash_valid,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# PRIME Voice Identity Manager
# =============================================================================

class PrimeVoiceIdentityManager:
    """
    Manages PRIME's locked voice identity.

    Ensures that PRIME's voice cannot be altered, spoofed, or drifted
    beyond acceptable tolerances. Acts as the guardian of PRIME's vocal DNA.

    Usage:
        manager = PrimeVoiceIdentityManager()
        identity = manager.get_identity()

        # Get voice parameters for synthesis
        voice_config = manager.get_synthesis_params()

        # Verify output matches PRIME's identity
        result = manager.verify_output(audio_embedding)

        # Check identity integrity
        check = manager.check_integrity()
    """

    def __init__(self, config: Optional[IdentityConfig] = None):
        self.config = config or IdentityConfig()
        self._identity = self._create_default_identity()
        self._integrity_hash = self._compute_integrity_hash()
        self._check_history: List[IdentityCheckResult] = []
        self._consecutive_failures = 0
        self._utterance_count = 0
        self._created_at = datetime.now()

        logger.info(
            f"PRIME Voice Identity initialized: {self._identity.identity_id} "
            f"[lock={self._identity.lock_level.value}]"
        )

    def _create_default_identity(self) -> PrimeVoiceIdentity:
        """Create PRIME's default voice identity."""
        return PrimeVoiceIdentity(
            identity_id="PRIME_VOICE_001",
            identity_name="PRIME",
            identity_version="1.0.0",
            lock_level=self.config.lock_level,
            voice_vector=PrimeVoiceVector(
                formality=0.6,       # Formal but not stiff
                temperature=0.4,     # Warm side of professional
                energy=-0.2,         # Calm, measured energy
                authority=0.7,       # Clearly authoritative
                abstraction=0.3,     # Concrete-leaning
                intimacy=-0.1,       # Professional distance
                certainty=0.6,       # Confident
                complexity=0.4,      # Adapts to audience
            ),
            vocal_dna=PrimeVocalDNA(
                target_pitch_hz=115.0,
                target_pitch_variance=0.3,
                target_speaking_rate=0.95,
                target_resonance=0.8,
                target_breathiness=0.1,
                target_vocal_fry=0.05,
                target_authority=0.85,
                target_warmth=0.6,
                target_trust=0.9,
                target_charisma=0.75,
                target_approachability=0.65,
                target_credibility=0.95,
                target_dominance=0.7,
                baseline_calm=0.85,
                baseline_confidence=0.9,
                baseline_focus=0.95,
                emotional_range=0.4,
            ),
            vox_voice_id="prime_sovereign",
            emotion_preset="professional",
            description="PRIME Sovereign Voice - All Signal, ZERO Noise",
        )

    def _compute_integrity_hash(self) -> str:
        """Compute cryptographic hash of the identity for tamper detection."""
        identity_data = json.dumps(self._identity.to_dict(), sort_keys=True)
        return hashlib.sha256(identity_data.encode()).hexdigest()

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------

    def get_identity(self) -> PrimeVoiceIdentity:
        """Get the current PRIME voice identity (read-only if locked)."""
        return self._identity

    def get_voice_vector(self) -> PrimeVoiceVector:
        """Get PRIME's 8-dimensional voice vector."""
        return self._identity.voice_vector

    def get_vocal_dna(self) -> PrimeVocalDNA:
        """Get PRIME's vocal DNA targets."""
        return self._identity.vocal_dna

    def get_synthesis_params(self) -> Dict[str, Any]:
        """
        Get parameters for VoxSynthesizer in PRIME's voice.

        Returns a dictionary compatible with VoiceConfig and ProsodyTarget.
        """
        dna = self._identity.vocal_dna
        vec = self._identity.voice_vector

        return {
            # VoiceConfig params
            "voice_id": self._identity.vox_voice_id,
            "speaking_rate": dna.target_speaking_rate,
            "pitch": 0.0,  # Base pitch (adjusted by speaking mode)
            "emotion": self._identity.emotion_preset,

            # Prosody params
            "warmth": dna.target_warmth,
            "confidence": dna.baseline_confidence,
            "energy": max(0.0, min(1.0, (vec.energy + 1.0) / 2.0)),

            # Voice vector (for VoiceSpaceDirector)
            "voice_vector": vec.to_dict(),

            # DNA targets (for advanced synthesis)
            "target_pitch_hz": dna.target_pitch_hz,
            "target_resonance": dna.target_resonance,
            "target_breathiness": dna.target_breathiness,
        }

    def get_voice_vector_list(self) -> List[float]:
        """Get the 8-dim vector as a list for VoiceVector compatibility."""
        return self._identity.voice_vector.to_list()

    # -------------------------------------------------------------------------
    # Identity Modification (Restricted)
    # -------------------------------------------------------------------------

    def calibrate(
        self,
        adjustments: Dict[str, float],
        authorization: str = "",
    ) -> bool:
        """
        Apply minor calibration adjustments to the voice identity.

        Only works when lock_level is CALIBRATING.
        Each adjustment is clamped to ±0.1 from current value.

        Args:
            adjustments: Dict of field_name -> new_value
            authorization: Authorization token for calibration

        Returns:
            True if calibration was applied
        """
        if self._identity.lock_level == IdentityLockLevel.LOCKED:
            logger.warning("PRIME voice identity is LOCKED - calibration rejected")
            return False

        if self._identity.lock_level != IdentityLockLevel.CALIBRATING:
            logger.warning(
                f"Calibration requires CALIBRATING mode, "
                f"current: {self._identity.lock_level.value}"
            )
            return False

        vec = self._identity.voice_vector
        max_delta = 0.1  # Maximum adjustment per calibration

        applied = []
        for field_name, target_value in adjustments.items():
            if hasattr(vec, field_name):
                current = getattr(vec, field_name)
                clamped = max(current - max_delta, min(current + max_delta, target_value))
                clamped = max(-1.0, min(1.0, clamped))
                setattr(vec, field_name, clamped)
                applied.append(f"{field_name}: {current:.3f} -> {clamped:.3f}")

        if applied:
            self._integrity_hash = self._compute_integrity_hash()
            logger.info(f"PRIME voice calibrated: {', '.join(applied)}")

        return bool(applied)

    def lock(self) -> None:
        """Lock the identity, preventing further modifications."""
        self._identity.lock_level = IdentityLockLevel.LOCKED
        self._integrity_hash = self._compute_integrity_hash()
        logger.info("PRIME voice identity LOCKED")

    # -------------------------------------------------------------------------
    # Biometric Anchor
    # -------------------------------------------------------------------------

    def enroll_biometric(self, embedding: List[float]) -> bool:
        """
        Register a biometric embedding as PRIME's voice anchor.

        This embedding is used to verify that synthesized output
        actually matches PRIME's expected voice signature.

        Args:
            embedding: Speaker embedding vector (e.g., 256-dim SpectralFingerprint)

        Returns:
            True if enrollment succeeded
        """
        if not embedding:
            logger.error("Cannot enroll empty embedding")
            return False

        self._identity.biometric_embedding = embedding
        self._identity.biometric_enrolled = True
        self._identity.enrollment_timestamp = datetime.now()
        self._integrity_hash = self._compute_integrity_hash()

        logger.info(
            f"PRIME biometric anchor enrolled: {len(embedding)}-dim embedding"
        )
        return True

    def verify_output(self, output_embedding: List[float]) -> IdentityCheckResult:
        """
        Verify that a synthesized audio output matches PRIME's voice.

        Compares the output embedding against the enrolled biometric anchor.

        Args:
            output_embedding: Embedding extracted from synthesized audio

        Returns:
            IdentityCheckResult with similarity and pass/fail
        """
        self._utterance_count += 1

        if not self._identity.biometric_enrolled or not self._identity.biometric_embedding:
            return IdentityCheckResult(
                passed=True,
                similarity_score=0.0,
                message="Biometric anchor not enrolled - verification skipped",
            )

        # Cosine similarity
        similarity = self._cosine_similarity(
            self._identity.biometric_embedding,
            output_embedding,
        )

        passed = similarity >= self.config.similarity_threshold

        result = IdentityCheckResult(
            passed=passed,
            similarity_score=similarity,
            tamper_suspected=not passed and similarity < 0.5,
            message=(
                f"Identity verified (similarity={similarity:.3f})"
                if passed
                else f"Identity mismatch (similarity={similarity:.3f}, "
                     f"threshold={self.config.similarity_threshold})"
            ),
        )

        # Track consecutive failures
        if passed:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.config.tamper_alert_threshold:
                result.tamper_suspected = True
                result.message += " - TAMPER ALERT: multiple consecutive failures"
                logger.error(
                    f"PRIME VOICE TAMPER ALERT: {self._consecutive_failures} "
                    f"consecutive identity failures"
                )

        self._check_history.append(result)

        # Trim history
        if len(self._check_history) > 100:
            self._check_history = self._check_history[-100:]

        return result

    # -------------------------------------------------------------------------
    # Integrity Checks
    # -------------------------------------------------------------------------

    def check_integrity(self) -> IdentityCheckResult:
        """
        Verify that the identity hasn't been tampered with.

        Compares current state against stored integrity hash.
        """
        current_hash = self._compute_integrity_hash()
        valid = current_hash == self._integrity_hash

        # Check for voice vector drift
        drift_dims = self._check_vector_drift()

        result = IdentityCheckResult(
            passed=valid and not drift_dims,
            integrity_hash_valid=valid,
            drift_detected=bool(drift_dims),
            drift_dimensions=drift_dims,
            tamper_suspected=not valid,
            message=(
                "Identity integrity verified"
                if valid and not drift_dims
                else f"Integrity issue: hash_valid={valid}, drift={drift_dims}"
            ),
        )

        if not valid:
            logger.error("PRIME voice identity integrity check FAILED - possible tampering")

        return result

    def _check_vector_drift(self) -> List[str]:
        """Check if voice vector has drifted from expected values."""
        expected = PrimeVoiceVector()  # Default PRIME values
        current = self._identity.voice_vector
        drifted = []

        for dim in [
            "formality", "temperature", "energy", "authority",
            "abstraction", "intimacy", "certainty", "complexity",
        ]:
            expected_val = getattr(expected, dim)
            current_val = getattr(current, dim)
            if abs(current_val - expected_val) > self.config.max_vector_drift:
                drifted.append(dim)

        return drifted

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    # -------------------------------------------------------------------------
    # Session Tracking
    # -------------------------------------------------------------------------

    def should_verify(self) -> bool:
        """Check if it's time for a biometric verification check."""
        if not self._identity.biometric_enrolled:
            return False
        return self._utterance_count % self.config.verification_interval == 0

    def get_stats(self) -> Dict[str, Any]:
        """Get identity manager statistics."""
        passed = sum(1 for c in self._check_history if c.passed)
        total = len(self._check_history)

        return {
            "identity_id": self._identity.identity_id,
            "lock_level": self._identity.lock_level.value,
            "biometric_enrolled": self._identity.biometric_enrolled,
            "utterance_count": self._utterance_count,
            "checks_total": total,
            "checks_passed": passed,
            "checks_failed": total - passed,
            "pass_rate": passed / total if total > 0 else 1.0,
            "consecutive_failures": self._consecutive_failures,
            "integrity_hash": self._integrity_hash[:16] + "...",
            "uptime_seconds": (datetime.now() - self._created_at).total_seconds(),
        }

    def get_identity_report(self) -> str:
        """Generate a markdown report of PRIME's voice identity."""
        stats = self.get_stats()
        identity = self._identity
        dna = identity.vocal_dna
        vec = identity.voice_vector

        lines = [
            "# PRIME Voice Identity Report",
            "",
            f"**Identity:** {identity.identity_id} v{identity.identity_version}",
            f"**Lock Level:** {identity.lock_level.value}",
            f"**Voice ID:** {identity.vox_voice_id}",
            f"**Biometric Enrolled:** {'Yes' if identity.biometric_enrolled else 'No'}",
            "",
            "## Voice Vector (8D Space)",
            "",
            f"| Dimension | Value | Description |",
            f"|-----------|-------|-------------|",
            f"| Formality | {vec.formality:+.2f} | {'Formal' if vec.formality > 0 else 'Casual'} |",
            f"| Temperature | {vec.temperature:+.2f} | {'Warm' if vec.temperature > 0 else 'Cool'} |",
            f"| Energy | {vec.energy:+.2f} | {'Energetic' if vec.energy > 0 else 'Calm'} |",
            f"| Authority | {vec.authority:+.2f} | {'Authoritative' if vec.authority > 0 else 'Supportive'} |",
            f"| Abstraction | {vec.abstraction:+.2f} | {'Abstract' if vec.abstraction > 0 else 'Concrete'} |",
            f"| Intimacy | {vec.intimacy:+.2f} | {'Intimate' if vec.intimacy > 0 else 'Distant'} |",
            f"| Certainty | {vec.certainty:+.2f} | {'Definitive' if vec.certainty > 0 else 'Tentative'} |",
            f"| Complexity | {vec.complexity:+.2f} | {'Complex' if vec.complexity > 0 else 'Simple'} |",
            "",
            "## Vocal DNA Targets",
            "",
            f"- Pitch: {dna.target_pitch_hz:.0f} Hz (variance: {dna.target_pitch_variance:.1%})",
            f"- Speaking Rate: {dna.target_speaking_rate:.2f}x",
            f"- Authority: {dna.target_authority:.0%}",
            f"- Warmth: {dna.target_warmth:.0%}",
            f"- Trust: {dna.target_trust:.0%}",
            f"- Charisma: {dna.target_charisma:.0%}",
            f"- Credibility: {dna.target_credibility:.0%}",
            "",
            "## Session Statistics",
            "",
            f"- Utterances: {stats['utterance_count']}",
            f"- Checks Passed: {stats['checks_passed']}/{stats['checks_total']}",
            f"- Pass Rate: {stats['pass_rate']:.0%}",
            f"- Consecutive Failures: {stats['consecutive_failures']}",
            "",
            f"*Integrity Hash: `{stats['integrity_hash']}`*",
        ]

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[PrimeVoiceIdentityManager] = None


def get_identity_manager(
    config: Optional[IdentityConfig] = None,
) -> PrimeVoiceIdentityManager:
    """Get or create the default PRIME voice identity manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PrimeVoiceIdentityManager(config)
    return _default_manager


def get_prime_identity() -> PrimeVoiceIdentity:
    """Quick access to PRIME's voice identity."""
    return get_identity_manager().get_identity()


def get_prime_synthesis_params() -> Dict[str, Any]:
    """Quick access to PRIME's synthesis parameters."""
    return get_identity_manager().get_synthesis_params()
