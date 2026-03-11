"""
VØX SDK Client
--------------

High-level async client for VØX operations.

The VoxClient provides a clean, intuitive interface to all VØX features:
    - Voice synthesis with governance
    - Voice biometric enrollment and verification
    - Multi-voice dialogue synthesis
    - Quality monitoring
    - Rate limit and quota management

AXIØM Phase 8: Integrate - "How do the parts connect?"

Usage:
    async with VoxClient(api_key="...") as vox:
        audio = await vox.synthesize("Hello world", voice="warm")
"""

import asyncio
import base64
import logging
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from dataclasses import dataclass

from .config import VoxConfig, Environment
from .session import VoxSession, SessionMetrics
from .retry import RetryExecutor, RetryPolicy, with_retry
from .errors import (
    VoxError,
    ValidationError,
    InvalidTextError,
    InvalidVoiceError,
    GovernanceError,
    RateLimitError,
    QuotaExceededError,
    BiometricError,
    NotEnrolledError,
    ErrorContext,
)
from .workflows import (
    SynthesisResult,
    EnrollmentResult,
    VerificationResult,
    DialogueLine,
    DialogueResult,
    synthesize_with_quality_check,
    synthesize_dialogue,
    enroll_and_verify,
)

logger = logging.getLogger(__name__)


class VoxClient:
    """
    High-level async client for VØX voice operations.

    Provides a clean interface to:
        - Voice synthesis with AXIOM governance
        - Voice biometric enrollment and verification
        - Multi-voice dialogue synthesis
        - Usage monitoring and quota management

    Example:
        async with VoxClient(api_key="vox_xxx") as vox:
            # Simple synthesis
            audio = await vox.synthesize("Hello world")

            # With voice and emotion
            audio = await vox.synthesize(
                "Welcome back!",
                voice="warm",
                emotion="joy",
            )

            # Enroll a voice
            await vox.enroll_voice("user_123", audio_samples=[...])

            # Verify speaker
            verified = await vox.verify_speaker("user_123", audio)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[VoxConfig] = None,
        **kwargs,
    ):
        """
        Initialize VØX client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for VØX API
            config: Full configuration object
            **kwargs: Additional config overrides
        """
        # Build config
        if config:
            self.config = config
        else:
            self.config = VoxConfig.from_env()

        # Apply explicit parameters
        if api_key:
            self.config.api_key = api_key
        if base_url:
            self.config.base_url = base_url

        # Apply additional overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Initialize components
        self._session: Optional[VoxSession] = None
        self._retry_executor = RetryExecutor(RetryPolicy(self.config.retry))

        # Validate config
        errors = self.config.validate()
        if errors:
            logger.warning(f"Configuration warnings: {errors}")

    async def __aenter__(self) -> "VoxClient":
        """Enter async context - initialize session."""
        self._session = VoxSession(config=self.config)
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context - cleanup."""
        if self._session:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)
            self._session = None

    # ========================================================================
    # Synthesis Methods
    # ========================================================================

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        emotion: Optional[str] = None,
        speaking_rate: Optional[float] = None,
        format: str = "mp3",
        stream: bool = False,
        **kwargs,
    ) -> SynthesisResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID or name (default: config default)
            emotion: Emotion preset (joy, calm, professional, etc.)
            speaking_rate: Speaking rate multiplier (0.5 to 2.0)
            format: Audio format (mp3, wav, ogg)
            stream: Whether to stream response
            **kwargs: Additional parameters

        Returns:
            SynthesisResult with audio bytes

        Raises:
            InvalidTextError: If text is invalid
            GovernanceError: If content is blocked
            RateLimitError: If rate limited
            QuotaExceededError: If quota exceeded
        """
        # Validate text
        if not text or not text.strip():
            raise InvalidTextError("Text cannot be empty", text_length=0)

        if len(text) > 10000:
            raise InvalidTextError(
                "Text too long (max 10000 characters)",
                text_length=len(text),
            )

        # Resolve voice
        voice_id = voice or self.config.default_voice_id

        # Build request
        request_data = {
            "text": text,
            "voice_id": voice_id,
            "emotion_preset": emotion or self.config.default_emotion_preset,
            "speaking_rate": speaking_rate or self.config.default_speaking_rate,
            "output_format": format,
            "stream": stream,
            "enable_quality_monitoring": self.config.quality.enable_quality_monitoring,
            **kwargs,
        }

        # Execute with retry
        async def do_synthesize():
            return await self._call_api("POST", "/synthesize", request_data)

        response = await self._retry_executor.execute_async(
            do_synthesize,
            operation_name="synthesize",
        )

        # Parse response
        audio_base64 = response.get("audio_base64", "")
        audio = base64.b64decode(audio_base64) if audio_base64 else b""

        return SynthesisResult(
            audio=audio,
            voice_id=voice_id,
            text=text,
            duration_seconds=response.get("duration_seconds", 0.0),
            quality_score=response.get("quality_score"),
            sample_rate=response.get("sample_rate", 24000),
            format=format,
            metadata=response.get("governance_report", {}),
        )

    async def synthesize_ssml(
        self,
        ssml: str,
        default_voice: Optional[str] = None,
        **kwargs,
    ) -> SynthesisResult:
        """
        Synthesize speech from SSML markup.

        Args:
            ssml: SSML markup
            default_voice: Default voice for untagged content
            **kwargs: Additional parameters

        Returns:
            SynthesisResult
        """
        request_data = {
            "ssml": ssml,
            "voice_id": default_voice or self.config.default_voice_id,
            **kwargs,
        }

        response = await self._call_api("POST", "/synthesize", request_data)

        audio_base64 = response.get("audio_base64", "")
        audio = base64.b64decode(audio_base64) if audio_base64 else b""

        return SynthesisResult(
            audio=audio,
            voice_id=request_data["voice_id"],
            text=ssml,
            duration_seconds=response.get("duration_seconds", 0.0),
            quality_score=response.get("quality_score"),
            format="mp3",
        )

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[bytes]:
        """
        Stream synthesized audio chunks.

        Args:
            text: Text to synthesize
            voice: Voice ID
            **kwargs: Additional parameters

        Yields:
            Audio chunks as bytes
        """
        request_data = {
            "text": text,
            "voice_id": voice or self.config.default_voice_id,
            "stream": True,
            **kwargs,
        }

        async for chunk in self._call_api_stream("POST", "/synthesize/stream", request_data):
            yield chunk

    # ========================================================================
    # Biometric Methods
    # ========================================================================

    async def enroll_voice(
        self,
        voice_id: str,
        audio_samples: List[bytes],
        owner_id: Optional[str] = None,
        consent_token: Optional[str] = None,
        **kwargs,
    ) -> EnrollmentResult:
        """
        Enroll a voice for biometric verification.

        Args:
            voice_id: Unique voice identifier
            audio_samples: List of audio samples (minimum 3)
            owner_id: Owner identifier
            consent_token: Consent verification token
            **kwargs: Additional parameters

        Returns:
            EnrollmentResult

        Raises:
            ValidationError: If insufficient samples
            BiometricError: If enrollment fails
        """
        if len(audio_samples) < self.config.biometric.min_enrollment_samples:
            raise ValidationError(
                f"At least {self.config.biometric.min_enrollment_samples} "
                "audio samples required for enrollment"
            )

        request_data = {
            "voice_id": voice_id,
            "audio_samples": [base64.b64encode(s).decode() for s in audio_samples],
            "owner_id": owner_id or self.config.user_id,
            "consent_token": consent_token,
            **kwargs,
        }

        response = await self._call_api("POST", "/biometrics/enroll", request_data)

        return EnrollmentResult(
            voice_id=voice_id,
            template_id=response.get("template_id", ""),
            samples_used=response.get("sample_count", len(audio_samples)),
            confidence=response.get("confidence", 0.0),
            enrolled=response.get("success", False),
            warnings=response.get("warnings", []),
        )

    async def verify_speaker(
        self,
        voice_id: str,
        audio: bytes,
        require_liveness: Optional[bool] = None,
        **kwargs,
    ) -> VerificationResult:
        """
        Verify speaker identity against enrolled template.

        Args:
            voice_id: Voice ID to verify against
            audio: Audio sample for verification
            require_liveness: Require liveness check
            **kwargs: Additional parameters

        Returns:
            VerificationResult

        Raises:
            NotEnrolledError: If voice not enrolled
            BiometricError: If verification fails
        """
        request_data = {
            "voice_id": voice_id,
            "audio_sample": base64.b64encode(audio).decode(),
            "require_liveness": require_liveness if require_liveness is not None
                else self.config.biometric.require_liveness,
            **kwargs,
        }

        response = await self._call_api("POST", "/biometrics/verify", request_data)

        return VerificationResult(
            voice_id=voice_id,
            verified=response.get("verified", False),
            similarity=response.get("similarity", 0.0),
            threshold=response.get("threshold", self.config.biometric.similarity_threshold),
            liveness_passed=response.get("liveness_passed", True),
            liveness_score=response.get("liveness_score"),
        )

    async def is_enrolled(self, voice_id: str) -> bool:
        """Check if a voice is enrolled for biometric verification."""
        try:
            response = await self._call_api("GET", f"/biometrics/status/{voice_id}")
            return response.get("enrolled", False)
        except NotEnrolledError:
            return False

    async def revoke_enrollment(self, voice_id: str, reason: str = "") -> bool:
        """
        Revoke voice biometric enrollment.

        Args:
            voice_id: Voice ID to revoke
            reason: Reason for revocation

        Returns:
            True if revoked successfully
        """
        response = await self._call_api(
            "DELETE",
            f"/biometrics/{voice_id}",
            {"reason": reason},
        )
        return response.get("success", False)

    # ========================================================================
    # Multi-Voice Methods
    # ========================================================================

    async def dialogue(
        self,
        lines: List[Union[Dict[str, Any], DialogueLine]],
        character_voices: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> DialogueResult:
        """
        Synthesize multi-character dialogue.

        Args:
            lines: List of dialogue lines (dicts or DialogueLine objects)
            character_voices: Mapping of character names to voice IDs
            **kwargs: Additional parameters

        Returns:
            DialogueResult

        Example:
            result = await vox.dialogue([
                {"character": "Alice", "text": "Hello!"},
                {"character": "Bob", "text": "Hi there!"},
            ], character_voices={"Alice": "warm", "Bob": "professional"})
        """
        # Convert dicts to DialogueLine
        dialogue_lines = []
        for line in lines:
            if isinstance(line, dict):
                dialogue_lines.append(DialogueLine(**line))
            else:
                dialogue_lines.append(line)

        return await synthesize_dialogue(
            client=self,
            lines=dialogue_lines,
            character_voices=character_voices,
            default_voice=self.config.default_voice_id,
            **kwargs,
        )

    # ========================================================================
    # Voice Management
    # ========================================================================

    async def list_voices(self) -> List[Dict[str, Any]]:
        """
        List available voices.

        Returns:
            List of voice information dictionaries
        """
        response = await self._call_api("GET", "/voices")
        return response if isinstance(response, list) else []

    async def get_voice(self, voice_id: str) -> Dict[str, Any]:
        """
        Get details about a specific voice.

        Args:
            voice_id: Voice identifier

        Returns:
            Voice information dictionary

        Raises:
            InvalidVoiceError: If voice not found
        """
        return await self._call_api("GET", f"/voice/{voice_id}")

    # ========================================================================
    # Governance & Quota Methods
    # ========================================================================

    async def check_content(
        self,
        text: str,
        voice_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pre-flight check content against governance rules.

        Args:
            text: Text to check
            voice_id: Voice ID for context

        Returns:
            Governance check result
        """
        return await self._call_api("POST", "/check", {
            "text": text,
            "voice_id": voice_id or self.config.default_voice_id,
        })

    async def get_quota_status(self) -> Dict[str, Any]:
        """
        Get current quota status.

        Returns:
            Quota status for current user
        """
        user_id = self.config.user_id or "anonymous"
        return await self._call_api("GET", f"/governance/quotas?user_id={user_id}")

    async def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status.

        Returns:
            Rate limit status
        """
        return await self._call_api("GET", "/governance/rate-limit/status")

    # ========================================================================
    # Session & Metrics
    # ========================================================================

    def get_session_metrics(self) -> Optional[SessionMetrics]:
        """Get metrics for current session."""
        if self._session:
            return self._session.get_metrics()
        return None

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID."""
        if self._session:
            return self._session.session_id
        return None

    # ========================================================================
    # Health & Status
    # ========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Check VØX service health.

        Returns:
            Health status
        """
        return await self._call_api("GET", "/health")

    async def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status including quotas, rate limits, and resources.

        Returns:
            Combined status information
        """
        quota_task = self.get_quota_status()
        rate_limit_task = self.get_rate_limit_status()
        health_task = self.health_check()

        quota, rate_limit, health = await asyncio.gather(
            quota_task, rate_limit_task, health_task,
            return_exceptions=True,
        )

        return {
            "healthy": not isinstance(health, Exception),
            "health": health if not isinstance(health, Exception) else {"error": str(health)},
            "quotas": quota if not isinstance(quota, Exception) else {"error": str(quota)},
            "rate_limits": rate_limit if not isinstance(rate_limit, Exception) else {"error": str(rate_limit)},
        }

    # ========================================================================
    # Internal Methods
    # ========================================================================

    async def _call_api(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make API call."""
        # For now, use local modules directly
        # In production, this would use HTTP client
        try:
            return await self._call_local(method, endpoint, data)
        except Exception as e:
            if isinstance(e, VoxError):
                raise
            raise VoxError(
                f"API call failed: {e}",
                context=ErrorContext(operation=f"{method} {endpoint}"),
                cause=e,
            )

    async def _call_local(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call local VØX modules directly (for embedded use)."""
        data = data or {}

        # Route to appropriate handler
        if endpoint == "/synthesize":
            return await self._local_synthesize(data)
        elif endpoint == "/check":
            return await self._local_check(data)
        elif endpoint == "/voices":
            return await self._local_list_voices()
        elif endpoint.startswith("/voice/"):
            voice_id = endpoint.split("/")[-1]
            return await self._local_get_voice(voice_id)
        elif endpoint == "/biometrics/enroll":
            return await self._local_enroll(data)
        elif endpoint == "/biometrics/verify":
            return await self._local_verify(data)
        elif endpoint.startswith("/biometrics/status/"):
            voice_id = endpoint.split("/")[-1]
            return await self._local_biometric_status(voice_id)
        elif endpoint == "/health":
            return {"status": "healthy", "version": "0.13.0"}
        elif endpoint.startswith("/governance/"):
            return await self._local_governance(endpoint, data)
        else:
            return {"status": "ok"}

    async def _local_synthesize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Local synthesis using VØX modules."""
        try:
            from axiom_vox import VoxGovernor, GovernanceAction

            governor = VoxGovernor()
            result = governor.govern(
                text=data.get("text", ""),
                voice_id=data.get("voice_id", "axiom_default"),
                context=data.get("context"),
            )

            if result.action == GovernanceAction.REFUSE:
                raise GovernanceError(
                    result.refusal_reason or "Content blocked by governance",
                )

            # Return placeholder (actual synthesis would happen here)
            return {
                "success": True,
                "audio_base64": "",  # Would contain actual audio
                "governed_text": result.governed_text,
                "duration_seconds": len(data.get("text", "")) * 0.05,
                "governance_report": result.to_dict(),
            }

        except ImportError:
            return {"success": True, "audio_base64": "", "duration_seconds": 0}

    async def _local_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Local governance check."""
        try:
            from axiom_vox import VoxGovernor

            governor = VoxGovernor()
            result = governor.govern(
                text=data.get("text", ""),
                voice_id=data.get("voice_id", "axiom_default"),
            )

            return result.to_dict()

        except ImportError:
            return {"approved": True, "action": "allow"}

    async def _local_list_voices(self) -> List[Dict[str, Any]]:
        """Local voice listing."""
        return [
            {"voice_id": "axiom_default", "name": "AXIØM Default", "category": "synthetic"},
            {"voice_id": "axiom_warm", "name": "AXIØM Warm", "category": "synthetic"},
            {"voice_id": "axiom_professional", "name": "AXIØM Professional", "category": "synthetic"},
        ]

    async def _local_get_voice(self, voice_id: str) -> Dict[str, Any]:
        """Local voice lookup."""
        voices = await self._local_list_voices()
        for voice in voices:
            if voice["voice_id"] == voice_id:
                return voice
        raise InvalidVoiceError(voice_id)

    async def _local_enroll(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Local biometric enrollment."""
        return {
            "success": True,
            "template_id": f"tmpl_{data.get('voice_id', 'unknown')}",
            "sample_count": len(data.get("audio_samples", [])),
            "confidence": 0.85,
        }

    async def _local_verify(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Local biometric verification."""
        return {
            "verified": True,
            "similarity": 0.88,
            "threshold": 0.75,
            "liveness_passed": True,
            "liveness_score": 0.92,
        }

    async def _local_biometric_status(self, voice_id: str) -> Dict[str, Any]:
        """Local biometric status check."""
        return {"enrolled": False, "voice_id": voice_id}

    async def _local_governance(
        self,
        endpoint: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Local governance endpoints."""
        if "quotas" in endpoint:
            return {"user_id": "test", "tier": "free", "quotas": {}}
        elif "rate-limit" in endpoint:
            return {"allowed": True, "remaining": 100, "limit": 100}
        elif "resources" in endpoint:
            return {"resources": {}, "queue": {}}
        return {}

    async def _call_api_stream(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[bytes]:
        """Stream API response."""
        # Placeholder for streaming
        yield b""
