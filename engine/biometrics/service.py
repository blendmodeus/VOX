"""
Voice Biometric Service
-----------------------

Main orchestrator for voice biometric verification.

Provides:
    - Voice enrollment with consent verification
    - Speaker verification with liveness detection
    - Template management (update, revoke)
    - Drift monitoring and alerts

AXIØM Phase 5: Resonance - "finding signature frequency"
"""

import io
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union

import numpy as np

from .models import (
    BiometricTemplate,
    BiometricConfig,
    BiometricAuditEntry,
    BiometricAction,
    EmbeddingBackend,
    EnrollmentResult,
    EnrollmentStatus,
    VerificationResult,
    VerificationStatus,
    LivenessResult,
    DriftReport,
    DriftSeverity,
)
from .embeddings import (
    EmbeddingExtractor,
    SpectralFingerprint,
    get_extractor,
    serialize_embedding,
    deserialize_embedding,
)
from .liveness import LivenessDetector, check_liveness
from .drift import DriftMonitor, AdaptiveTemplateUpdater, analyze_drift
from .storage import BiometricStorage, get_biometric_storage
from .consent import (
    BiometricConsentManager,
    ConsentType,
    ConsentStatus,
    get_consent_manager,
)

logger = logging.getLogger(__name__)


def audio_bytes_to_array(audio_bytes: bytes, sample_rate: int = 24000) -> np.ndarray:
    """Convert audio bytes to numpy array."""
    try:
        import wave

        with io.BytesIO(audio_bytes) as bio:
            with wave.open(bio, 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                audio = audio / 32768.0  # Normalize to [-1, 1]
                return audio
    except Exception:
        # Try raw float32 bytes
        try:
            return np.frombuffer(audio_bytes, dtype=np.float32)
        except Exception:
            # Try raw int16 bytes
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            return audio / 32768.0


class VoiceBiometricService:
    """
    Main service for voice biometric operations.

    Orchestrates:
        - Embedding extraction
        - Liveness detection
        - Template storage
        - Drift monitoring
        - Consent verification
    """

    def __init__(
        self,
        config: Optional[BiometricConfig] = None,
        storage: Optional[BiometricStorage] = None,
        consent_manager: Optional[BiometricConsentManager] = None,
    ):
        """
        Initialize biometric service.

        Args:
            config: Service configuration
            storage: Biometric storage instance
            consent_manager: Consent manager instance
        """
        self.config = config or BiometricConfig()
        self._storage = storage
        self._consent_manager = consent_manager

        # Initialize components
        self.extractor = get_extractor(self.config.embedding_backend)
        self.liveness_detector = LivenessDetector(
            threshold=self.config.liveness_threshold,
        )
        self.drift_monitor = DriftMonitor(
            short_term_threshold=self.config.short_term_drift_threshold,
            long_term_threshold=self.config.long_term_drift_threshold,
        )
        self.template_updater = AdaptiveTemplateUpdater()

    @property
    def storage(self) -> BiometricStorage:
        """Get storage, lazy loading if needed."""
        if self._storage is None:
            self._storage = get_biometric_storage()
        return self._storage

    @property
    def consent_manager(self) -> BiometricConsentManager:
        """Get consent manager, lazy loading if needed."""
        if self._consent_manager is None:
            self._consent_manager = get_consent_manager()
        return self._consent_manager

    async def enroll(
        self,
        voice_id: str,
        audio_samples: List[bytes],
        owner_id: str,
        consent_token: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EnrollmentResult:
        """
        Enroll a voice with biometric template.

        Args:
            voice_id: Unique voice identifier
            audio_samples: List of audio samples (min 3, ~5s each)
            owner_id: Owner ID (for consent verification)
            consent_token: Consent token for enrollment
            metadata: Additional metadata

        Returns:
            EnrollmentResult
        """
        start_time = time.time()

        try:
            # Check if already enrolled
            if self.storage.is_enrolled(voice_id):
                return EnrollmentResult(
                    status=EnrollmentStatus.ALREADY_ENROLLED,
                    voice_id=voice_id,
                    message="Voice is already enrolled",
                )

            # Verify consent
            consent_result = self.consent_manager.require_consent(
                voice_id,
                consent_token,
                ConsentType.ENROLLMENT,
            )

            if consent_result.status != ConsentStatus.VALID:
                # Record audit
                self._audit_action(
                    voice_id=voice_id,
                    action=BiometricAction.ENROLL,
                    result="blocked",
                    error_message=f"Consent: {consent_result.message}",
                )

                return EnrollmentResult(
                    status=EnrollmentStatus.CONSENT_REQUIRED,
                    voice_id=voice_id,
                    message=consent_result.message,
                )

            # Validate samples
            if len(audio_samples) < self.config.min_enrollment_samples:
                return EnrollmentResult(
                    status=EnrollmentStatus.INSUFFICIENT_SAMPLES,
                    voice_id=voice_id,
                    sample_count=len(audio_samples),
                    message=f"Need at least {self.config.min_enrollment_samples} samples",
                )

            # Process samples
            embeddings = []
            quality_scores = []
            warnings = []

            for i, sample_bytes in enumerate(audio_samples[:self.config.max_enrollment_samples]):
                try:
                    audio = audio_bytes_to_array(sample_bytes)

                    # Check duration
                    duration = len(audio) / 24000  # Assume 24kHz
                    if duration < self.config.min_sample_duration:
                        warnings.append(f"Sample {i+1} too short: {duration:.1f}s")
                        continue

                    # Check liveness
                    if self.config.require_liveness:
                        liveness = self.liveness_detector.check(audio, 24000)
                        if not liveness.passed:
                            warnings.append(f"Sample {i+1} failed liveness: {liveness.message}")
                            continue

                    # Extract embedding
                    result = self.extractor.extract_and_serialize(audio, 24000)
                    if not result.is_valid:
                        warnings.append(f"Sample {i+1} extraction failed: {result.message}")
                        continue

                    embeddings.append(deserialize_embedding(result.embedding))
                    quality_scores.append(result.quality_score)

                except Exception as e:
                    warnings.append(f"Sample {i+1} error: {str(e)}")
                    continue

            # Check we have enough valid samples
            if len(embeddings) < self.config.min_enrollment_samples:
                self._audit_action(
                    voice_id=voice_id,
                    action=BiometricAction.ENROLL,
                    result="failure",
                    error_message=f"Only {len(embeddings)} valid samples",
                )

                return EnrollmentResult(
                    status=EnrollmentStatus.LOW_QUALITY,
                    voice_id=voice_id,
                    sample_count=len(embeddings),
                    quality_scores=quality_scores,
                    warnings=warnings,
                    message=f"Only {len(embeddings)} valid samples, need {self.config.min_enrollment_samples}",
                )

            # Average embeddings to create template
            template_embedding = np.mean(embeddings, axis=0)

            # L2 normalize
            norm = np.linalg.norm(template_embedding)
            if norm > 0:
                template_embedding = template_embedding / norm

            # Calculate confidence (inter-sample consistency)
            similarities = []
            for emb in embeddings:
                sim = float(np.dot(template_embedding, emb / np.linalg.norm(emb)))
                similarities.append(sim)
            confidence = float(np.mean(similarities))

            # Create template
            template = BiometricTemplate(
                template_id=f"bio_{uuid.uuid4().hex[:12]}",
                voice_id=voice_id,
                embedding=serialize_embedding(template_embedding),
                embedding_version="1.0",
                embedding_backend=self.config.embedding_backend,
                embedding_dim=self.config.embedding_dim,
                sample_count=len(embeddings),
                confidence=confidence,
                consent_id=consent_result.consent_id,
                owner_id=owner_id,
                enrolled_at=time.time(),
                metadata=metadata or {},
            )

            # Save template
            template_id = self.storage.save_template(template)

            # Record consent if token was provided
            if consent_token:
                self.consent_manager.record_consent(
                    voice_id=voice_id,
                    consent_type=ConsentType.ENROLLMENT,
                    owner_id=owner_id,
                    proof=consent_token[:64],
                    metadata={"template_id": template_id},
                )

            # Audit
            self._audit_action(
                voice_id=voice_id,
                action=BiometricAction.ENROLL,
                result="success",
                similarity_score=confidence,
                metadata={
                    "sample_count": len(embeddings),
                    "template_id": template_id,
                    "duration_ms": (time.time() - start_time) * 1000,
                },
            )

            return EnrollmentResult(
                status=EnrollmentStatus.SUCCESS,
                template_id=template_id,
                voice_id=voice_id,
                embedding_dim=self.config.embedding_dim,
                sample_count=len(embeddings),
                confidence=confidence,
                quality_scores=quality_scores,
                average_quality=float(np.mean(quality_scores)) if quality_scores else 0.0,
                warnings=warnings,
                message="Enrollment successful",
                enrolled_at=template.enrolled_at,
            )

        except Exception as e:
            logger.error(f"Enrollment error: {e}")
            self._audit_action(
                voice_id=voice_id,
                action=BiometricAction.ENROLL,
                result="failure",
                error_message=str(e),
            )

            return EnrollmentResult(
                status=EnrollmentStatus.ERROR,
                voice_id=voice_id,
                message=f"Enrollment error: {str(e)}",
            )

    async def verify(
        self,
        voice_id: str,
        audio_sample: bytes,
        require_liveness: bool = True,
    ) -> VerificationResult:
        """
        Verify audio against enrolled template.

        Args:
            voice_id: Voice ID to verify against
            audio_sample: Audio sample bytes
            require_liveness: Whether to require liveness check

        Returns:
            VerificationResult
        """
        start_time = time.time()

        try:
            # Get template
            template = self.storage.get_template(voice_id)

            if template is None:
                return VerificationResult(
                    status=VerificationStatus.NOT_ENROLLED,
                    voice_id=voice_id,
                    message="Voice not enrolled",
                )

            if template.revoked:
                return VerificationResult(
                    status=VerificationStatus.TEMPLATE_REVOKED,
                    voice_id=voice_id,
                    message="Template has been revoked",
                )

            # Convert audio
            audio = audio_bytes_to_array(audio_sample)

            # Liveness check
            liveness_result = None
            if require_liveness or self.config.require_liveness:
                liveness_result = self.liveness_detector.check(audio, 24000)

                if not liveness_result.passed:
                    self._audit_action(
                        voice_id=voice_id,
                        action=BiometricAction.VERIFY,
                        result="failure",
                        liveness_score=liveness_result.overall_score,
                        liveness_passed=False,
                        error_message=liveness_result.message,
                    )

                    return VerificationResult(
                        status=VerificationStatus.LIVENESS_FAILED,
                        voice_id=voice_id,
                        liveness_passed=False,
                        liveness_score=liveness_result.overall_score,
                        liveness_details={
                            "replay_detected": liveness_result.replay_detected,
                            "deepfake_detected": liveness_result.deepfake_detected,
                        },
                        message=liveness_result.message,
                    )

            # Extract embedding
            extract_result = self.extractor.extract_and_serialize(audio, 24000)
            if not extract_result.is_valid:
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    voice_id=voice_id,
                    message=f"Embedding extraction failed: {extract_result.message}",
                )

            # Compare embeddings
            sample_embedding = deserialize_embedding(extract_result.embedding)
            template_embedding = deserialize_embedding(template.embedding)

            similarity = self.extractor.similarity(sample_embedding, template_embedding)

            # Determine if verified
            is_verified = similarity >= self.config.similarity_threshold

            # Record to history (for drift analysis)
            self.storage.save_embedding_history(
                voice_id=voice_id,
                embedding=extract_result.embedding,
                context="verification",
                similarity_to_template=similarity,
                quality_score=extract_result.quality_score,
            )

            # Check drift
            history = self.storage.get_embedding_history(voice_id, limit=50)
            drift_report = self.drift_monitor.analyze(voice_id, history)

            # Determine final status
            if is_verified:
                status = VerificationStatus.VERIFIED
                message = "Verification successful"
            else:
                status = VerificationStatus.REJECTED
                message = f"Similarity {similarity:.3f} below threshold {self.config.similarity_threshold}"

            # Audit
            self._audit_action(
                voice_id=voice_id,
                action=BiometricAction.VERIFY,
                result="success" if is_verified else "failure",
                similarity_score=similarity,
                liveness_score=liveness_result.overall_score if liveness_result else None,
                liveness_passed=liveness_result.passed if liveness_result else None,
                drift_detected=drift_report.severity != DriftSeverity.NONE,
                metadata={
                    "duration_ms": (time.time() - start_time) * 1000,
                    "drift_severity": drift_report.severity.value,
                },
            )

            return VerificationResult(
                status=status,
                voice_id=voice_id,
                similarity_score=similarity,
                threshold=self.config.similarity_threshold,
                is_verified=is_verified,
                liveness_passed=liveness_result.passed if liveness_result else True,
                liveness_score=liveness_result.overall_score if liveness_result else 1.0,
                liveness_details=liveness_result.details if liveness_result else {},
                drift_detected=drift_report.severity != DriftSeverity.NONE,
                drift_severity=drift_report.severity,
                confidence=template.confidence,
                message=message,
            )

        except Exception as e:
            logger.error(f"Verification error: {e}")
            self._audit_action(
                voice_id=voice_id,
                action=BiometricAction.VERIFY,
                result="failure",
                error_message=str(e),
            )

            return VerificationResult(
                status=VerificationStatus.ERROR,
                voice_id=voice_id,
                message=f"Verification error: {str(e)}",
            )

    async def check_drift(self, voice_id: str) -> DriftReport:
        """
        Analyze voice drift for an enrolled voice.

        Args:
            voice_id: Voice ID

        Returns:
            DriftReport
        """
        template = self.storage.get_template(voice_id)
        if template is None:
            return DriftReport(
                voice_id=voice_id,
                severity=DriftSeverity.NONE,
                message="Voice not enrolled",
            )

        history = self.storage.get_embedding_history(voice_id, limit=100)
        report = self.drift_monitor.analyze(voice_id, history, template.embedding)

        # Audit
        self._audit_action(
            voice_id=voice_id,
            action=BiometricAction.DRIFT_CHECK,
            result="success",
            drift_detected=report.severity != DriftSeverity.NONE,
            metadata={
                "severity": report.severity.value,
                "short_term_drift": report.short_term_drift,
                "long_term_drift": report.long_term_drift,
            },
        )

        return report

    async def update_template(
        self,
        voice_id: str,
        audio_samples: List[bytes],
        consent_token: Optional[str] = None,
    ) -> EnrollmentResult:
        """
        Update template with new samples.

        Args:
            voice_id: Voice ID
            audio_samples: New audio samples
            consent_token: Consent token for update

        Returns:
            EnrollmentResult
        """
        # Verify consent for update
        consent_result = self.consent_manager.require_consent(
            voice_id,
            consent_token,
            ConsentType.UPDATE,
        )

        if consent_result.status != ConsentStatus.VALID:
            return EnrollmentResult(
                status=EnrollmentStatus.CONSENT_REQUIRED,
                voice_id=voice_id,
                message=consent_result.message,
            )

        # Get existing template
        template = self.storage.get_template(voice_id)
        if template is None:
            return EnrollmentResult(
                status=EnrollmentStatus.ERROR,
                voice_id=voice_id,
                message="Voice not enrolled",
            )

        # Extract new embeddings
        embeddings = []
        quality_scores = []

        for sample_bytes in audio_samples:
            try:
                audio = audio_bytes_to_array(sample_bytes)
                result = self.extractor.extract_and_serialize(audio, 24000)
                if result.is_valid:
                    embeddings.append(deserialize_embedding(result.embedding))
                    quality_scores.append(result.quality_score)
            except Exception:
                continue

        if not embeddings:
            return EnrollmentResult(
                status=EnrollmentStatus.LOW_QUALITY,
                voice_id=voice_id,
                message="No valid samples in update",
            )

        # Compute updated embedding
        template_embedding = deserialize_embedding(template.embedding)
        new_embedding = np.mean(embeddings, axis=0)
        new_embedding = new_embedding / np.linalg.norm(new_embedding)

        # Check similarity to existing template
        similarity = self.extractor.similarity(new_embedding, template_embedding)

        # Use adaptive updater
        drift_report = await self.check_drift(voice_id)

        if not self.template_updater.should_update(similarity, len(embeddings), drift_report):
            return EnrollmentResult(
                status=EnrollmentStatus.ERROR,
                voice_id=voice_id,
                message="Update rejected: samples too different from template or anomaly detected",
            )

        # Compute weighted update
        updated_embedding = self.template_updater.compute_updated_embedding(
            template_embedding,
            new_embedding,
            similarity,
        )

        # Update in storage
        new_sample_count = template.sample_count + len(embeddings)
        new_confidence = (template.confidence * template.sample_count + similarity * len(embeddings)) / new_sample_count

        self.storage.update_template(
            voice_id=voice_id,
            embedding=serialize_embedding(updated_embedding),
            sample_count=new_sample_count,
            confidence=new_confidence,
        )

        # Audit
        self._audit_action(
            voice_id=voice_id,
            action=BiometricAction.UPDATE,
            result="success",
            similarity_score=similarity,
            metadata={
                "samples_added": len(embeddings),
                "new_confidence": new_confidence,
            },
        )

        return EnrollmentResult(
            status=EnrollmentStatus.SUCCESS,
            template_id=template.template_id,
            voice_id=voice_id,
            sample_count=new_sample_count,
            confidence=new_confidence,
            quality_scores=quality_scores,
            average_quality=float(np.mean(quality_scores)),
            message="Template updated successfully",
        )

    async def revoke_template(
        self,
        voice_id: str,
        reason: str = "",
    ) -> bool:
        """
        Revoke a biometric template.

        Args:
            voice_id: Voice ID
            reason: Reason for revocation

        Returns:
            True if revoked
        """
        revoked = self.storage.revoke_template(voice_id, reason)

        if revoked:
            # Also revoke consent
            self.consent_manager.revoke_consent(voice_id, reason=reason)

            # Audit
            self._audit_action(
                voice_id=voice_id,
                action=BiometricAction.REVOKE,
                result="success",
                metadata={"reason": reason},
            )

        return revoked

    async def get_status(self, voice_id: str) -> Dict[str, Any]:
        """
        Get enrollment status for a voice.

        Args:
            voice_id: Voice ID

        Returns:
            Status dictionary
        """
        template = self.storage.get_template(voice_id)

        if template is None:
            return {
                "voice_id": voice_id,
                "enrolled": False,
            }

        # Get verification stats
        stats = self.storage.get_verification_stats(voice_id)

        return {
            "voice_id": voice_id,
            "enrolled": True,
            "template_id": template.template_id,
            "sample_count": template.sample_count,
            "confidence": template.confidence,
            "enrolled_at": template.enrolled_at,
            "updated_at": template.updated_at,
            "is_active": template.is_active,
            "verification_stats": stats,
        }

    def _audit_action(
        self,
        voice_id: str,
        action: BiometricAction,
        result: str,
        similarity_score: Optional[float] = None,
        liveness_score: Optional[float] = None,
        liveness_passed: Optional[bool] = None,
        drift_detected: bool = False,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record action to audit log."""
        if not self.config.audit_all_operations:
            return

        entry = BiometricAuditEntry(
            timestamp=time.time(),
            voice_id=voice_id,
            action=action,
            result=result,
            similarity_score=similarity_score,
            liveness_score=liveness_score,
            liveness_passed=liveness_passed,
            drift_detected=drift_detected,
            error_message=error_message,
            metadata=metadata or {},
        )

        try:
            self.storage.log_biometric_action(entry)
        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")


# Singleton instance
_service_instance: Optional[VoiceBiometricService] = None


def get_biometric_service(
    config: Optional[BiometricConfig] = None,
) -> VoiceBiometricService:
    """Get or create biometric service singleton."""
    global _service_instance
    if _service_instance is None:
        _service_instance = VoiceBiometricService(config=config)
    return _service_instance


def set_biometric_service(service: VoiceBiometricService) -> None:
    """Set the biometric service singleton."""
    global _service_instance
    _service_instance = service


# Convenience functions
async def enroll_voice(
    voice_id: str,
    audio_samples: List[bytes],
    owner_id: str,
    consent_token: Optional[str] = None,
) -> EnrollmentResult:
    """Enroll a voice with biometric verification."""
    service = get_biometric_service()
    return await service.enroll(voice_id, audio_samples, owner_id, consent_token)


async def verify_voice(
    voice_id: str,
    audio_sample: bytes,
    require_liveness: bool = True,
) -> VerificationResult:
    """Verify audio against enrolled voice."""
    service = get_biometric_service()
    return await service.verify(voice_id, audio_sample, require_liveness)
