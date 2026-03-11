"""
Governed TTS API
----------------

FastAPI wrapper that provides a governed text-to-speech endpoint.
Refuses to synthesize speech that fails AXIØM governance.

Architecture:
    Client Request → [Auth] → [VoxGovernor] → [VØX TTS] → Audio Response
                                    ↓
                              [Refuse/Repair]

Endpoints:
    POST /synthesize         - Governed TTS synthesis
    POST /synthesize/stream  - Streaming TTS synthesis (HTTP)
    POST /check              - Pre-flight governance check
    GET  /voices             - List available voices
    GET  /voice/{id}         - Voice details and clearance status
    POST /clone              - Request voice clone (with ethics check)
    GET  /health             - Service health
    WS   /ws/synthesize      - WebSocket streaming TTS

    Fine-tuning:
    POST   /finetune         - Start fine-tuning job
    GET    /finetune         - List fine-tuning jobs
    GET    /finetune/{id}    - Get job status
    DELETE /finetune/{id}    - Cancel job

Features:
    - Full AXIØM governance pipeline
    - Voice boundary enforcement
    - Prosody guardrails
    - Audit logging
    - Rate limiting
    - API key authentication

Usage:
    from axiom_vox import create_governed_tts_app

    app = create_governed_tts_app()

    # Run with: uvicorn governed_tts_api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import hashlib
import json

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    import asyncio
    import struct
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# AXIØM VØX imports
from axiom_vox.vox_governor import VoxGovernor, VoxGovernanceResult, GovernanceAction
from axiom_vox.voice_boundaries import (
    VoiceBoundaries, VoiceCloneRequest, VoiceCategory
)
from axiom_vox.prosody_guardrails import (
    ProsodyGuardrails, EmotionalIntent, EmotionCategory
)

# Fine-tuning imports (lazy loaded to avoid circular imports)
_job_manager = None


def _get_job_manager():
    """Get or create the fine-tuning job manager."""
    global _job_manager
    if _job_manager is None:
        try:
            from axiom_vox.finetuning import FineTuningJobManager
            from axiom_vox.persistence import get_database
            db = get_database()
            _job_manager = FineTuningJobManager(db=db, max_concurrent_jobs=2)
        except ImportError as e:
            logger.warning(f"Fine-tuning module not available: {e}")
            return None
    return _job_manager

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

if HAS_FASTAPI:

    class SynthesizeRequest(BaseModel):
        """Request to synthesize speech."""
        # Text input (provide text OR ssml, not both)
        text: Optional[str] = Field(None, description="Plain text to synthesize", max_length=5000)
        ssml: Optional[str] = Field(None, description="SSML markup to synthesize", max_length=10000)

        # Voice selection
        voice_id: str = Field(default="axiom_default", description="Voice to use")

        # Emotion control (preset name OR explicit values)
        emotion_preset: Optional[str] = Field(None, description="Named emotion preset (joy, sadness, calm, etc.)")
        emotion: Optional[str] = Field(None, description="Target emotion category (legacy)")

        # Fine-grained emotion control (overrides preset if specified)
        valence: Optional[float] = Field(None, ge=-1, le=1, description="Emotional valence")
        arousal: Optional[float] = Field(None, ge=0, le=1, description="Arousal level")
        dominance: Optional[float] = Field(None, ge=0, le=1, description="Dominance level")
        warmth: Optional[float] = Field(None, ge=0, le=1, description="Warmth level")
        confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence level")

        # Prosody control
        speaking_rate: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="Speed")

        # Output options
        output_format: str = Field("mp3", description="Audio format (mp3, wav, ogg)")
        stream: bool = Field(False, description="Stream audio chunks")

        # Context
        user_prompt: Optional[str] = Field(None, description="Original user request")
        context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    class SynthesizeResponse(BaseModel):
        """Response from synthesis."""
        success: bool
        action: str
        governed_text: str
        audio_url: Optional[str] = None
        audio_base64: Optional[str] = None
        duration_seconds: Optional[float] = None
        governance_report: Dict[str, Any]
        warnings: List[str] = []

    class CheckRequest(BaseModel):
        """Pre-flight governance check request."""
        text: str = Field(..., description="Text to check")
        voice_id: str = Field(default="axiom_default")
        emotion: Optional[str] = None
        context: Optional[Dict[str, Any]] = None

    class CheckResponse(BaseModel):
        """Response from governance check."""
        would_pass: bool
        action: str
        reason: str
        repairs_needed: List[str] = []
        warnings: List[str] = []

    class CloneRequest(BaseModel):
        """Request to clone a voice."""
        voice_name: str = Field(..., description="Name for the cloned voice")
        audio_samples: List[str] = Field(..., description="Base64 audio samples")
        intended_use: str = Field("general", description="Intended use case")
        consent_proof: Optional[str] = Field(None, description="Consent documentation")
        owner_id: Optional[str] = Field(None, description="Voice owner ID")

    class CloneResponse(BaseModel):
        """Response from clone request."""
        approved: bool
        voice_id: Optional[str] = None
        reason: str
        required_disclaimers: List[str] = []
        usage_restrictions: List[str] = []

    class VoiceInfo(BaseModel):
        """Information about a voice."""
        voice_id: str
        name: str
        category: str
        description: Optional[str] = None
        sample_url: Optional[str] = None
        languages: List[str] = ["en"]
        is_cloned: bool = False
        consent_verified: bool = False
        allowed_uses: List[str] = ["general"]

    # ========================================================================
    # FINE-TUNING MODELS
    # ========================================================================

    class FineTuneRequest(BaseModel):
        """Request to start a voice fine-tuning job."""
        voice_name: str = Field(..., description="Name for the fine-tuned voice")
        audio_files: List[str] = Field(..., description="Base64-encoded audio samples")
        consent_verified: bool = Field(False, description="Whether consent is verified")
        intended_use: str = Field("personal", description="Intended use case")
        owner_id: Optional[str] = Field(None, description="Voice owner identifier")
        epochs: int = Field(50, ge=10, le=200, description="Training epochs")
        fast_mode: bool = Field(False, description="Use fast training configuration")

    class FineTuneResponse(BaseModel):
        """Response from fine-tuning job creation."""
        success: bool
        job_id: Optional[str] = None
        voice_id: Optional[str] = None
        status: str
        message: str
        estimated_duration_minutes: Optional[float] = None

    class JobStatusResponse(BaseModel):
        """Response from job status query."""
        job_id: str
        voice_id: str
        status: str
        progress: float = 0.0
        current_epoch: int = 0
        total_epochs: int = 0
        created_at: str
        started_at: Optional[str] = None
        completed_at: Optional[str] = None
        estimated_remaining_seconds: Optional[float] = None
        final_loss: Optional[float] = None
        similarity_score: Optional[float] = None
        quality_score: Optional[float] = None
        verification_passed: Optional[bool] = None
        error_message: Optional[str] = None

    class JobCancelResponse(BaseModel):
        """Response from job cancellation."""
        success: bool
        job_id: str
        message: str

    # ========================================================================
    # STREAMING MODELS
    # ========================================================================

    class StreamSynthesizeRequest(BaseModel):
        """Request for streaming synthesis."""
        text: str = Field(..., description="Text to synthesize", max_length=10000)
        voice_id: str = Field(default="axiom_default", description="Voice to use")
        emotion: Optional[str] = Field(None, description="Target emotion")
        arousal: Optional[float] = Field(None, ge=0, le=1)
        valence: Optional[float] = Field(None, ge=-1, le=1)
        speaking_rate: Optional[float] = Field(1.0, ge=0.5, le=2.0)
        context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
        chunk_size: int = Field(4096, ge=1024, le=16384, description="Audio chunk size in bytes")
        include_timings: bool = Field(True, description="Include timing metadata")

    class StreamStartedResponse(BaseModel):
        """Response when stream starts."""
        request_id: str
        status: str = "started"
        total_sentences: int
        sample_rate: int = 24000
        format: str = "wav"
        governance_action: str
        governance_report: Optional[Dict[str, Any]] = None

    # ========================================================================
    # MULTI-VOICE MODELS (v0.9.0)
    # ========================================================================

    class DialogueLineRequest(BaseModel):
        """A single line of dialogue."""
        text: str = Field(..., description="Text to synthesize")
        voice_id: str = Field(default="professional", description="Voice to use")
        character_name: Optional[str] = Field(None, description="Character name for registry lookup")
        emotion: Optional[str] = Field(None, description="Emotion preset")
        pause_before_ms: int = Field(0, ge=0, le=5000, description="Pause before line in ms")
        pause_after_ms: int = Field(0, ge=0, le=5000, description="Pause after line in ms")

    class DialogueScriptRequest(BaseModel):
        """Structured dialogue script for multi-voice synthesis."""
        lines: List[DialogueLineRequest] = Field(..., min_length=1, max_length=100)
        default_transition: str = Field("breath_pause", description="Transition style between voices")

    class MultiVoiceSynthesizeRequest(BaseModel):
        """Request for multi-voice synthesis (v0.9.0)."""
        # Input format (provide one)
        ssml: Optional[str] = Field(None, description="SSML with <voice> tags", max_length=20000)
        script: Optional[DialogueScriptRequest] = Field(None, description="Structured dialogue script")

        # Default settings
        default_voice_id: str = Field("professional", description="Default voice for untagged content")

        # Output options
        output_format: str = Field("wav", description="Audio format (wav, mp3)")
        stream: bool = Field(False, description="Stream audio chunks")

        # Context
        context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    class VoiceSwitchInfo(BaseModel):
        """Information about a voice switch."""
        from_voice: str
        to_voice: str
        transition_style: str
        word_index: int
        character_from: Optional[str] = None
        character_to: Optional[str] = None

    class SegmentInfo(BaseModel):
        """Information about a synthesized segment."""
        text: str
        voice_id: str
        character_name: Optional[str] = None
        start_ms: float
        end_ms: float
        duration_ms: float

    class MultiVoiceSynthesizeResponse(BaseModel):
        """Response from multi-voice synthesis."""
        success: bool
        total_duration_ms: float
        synthesis_time_ms: float
        segments: List[SegmentInfo]
        voice_switches: List[VoiceSwitchInfo]
        voices_used: List[str]
        audio_base64: Optional[str] = None
        audio_url: Optional[str] = None
        errors: Optional[List[str]] = None

    # ========================================================================
    # BIOMETRIC MODELS (v0.10.0)
    # ========================================================================

    class BiometricEnrollRequest(BaseModel):
        """Request to enroll voice biometrics."""
        voice_id: str = Field(..., description="Unique voice identifier")
        audio_samples: List[str] = Field(..., description="Base64-encoded audio samples (min 3)")
        owner_id: str = Field(..., description="Owner identifier")
        consent_token: Optional[str] = Field(None, description="Consent token for enrollment")
        metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class BiometricEnrollResponse(BaseModel):
        """Response from biometric enrollment."""
        success: bool
        status: str
        template_id: Optional[str] = None
        voice_id: Optional[str] = None
        sample_count: int = 0
        confidence: float = 0.0
        message: str = ""
        warnings: List[str] = []

    class BiometricVerifyRequest(BaseModel):
        """Request to verify voice biometrics."""
        voice_id: str = Field(..., description="Voice ID to verify against")
        audio_sample: str = Field(..., description="Base64-encoded audio sample")
        require_liveness: bool = Field(True, description="Require liveness check")

    class BiometricVerifyResponse(BaseModel):
        """Response from biometric verification."""
        verified: bool
        status: str
        voice_id: str
        similarity_score: float = 0.0
        threshold: float = 0.75
        liveness_passed: bool = False
        liveness_score: float = 0.0
        drift_detected: bool = False
        drift_severity: str = "none"
        message: str = ""

    class BiometricStatusResponse(BaseModel):
        """Response for enrollment status."""
        voice_id: str
        enrolled: bool
        template_id: Optional[str] = None
        sample_count: int = 0
        confidence: float = 0.0
        enrolled_at: Optional[float] = None
        updated_at: Optional[float] = None
        verification_stats: Optional[Dict[str, Any]] = None

    class BiometricDriftResponse(BaseModel):
        """Response for drift analysis."""
        voice_id: str
        severity: str
        short_term_drift: float = 0.0
        long_term_drift: float = 0.0
        trend_direction: str = "stable"
        requires_re_enrollment: bool = False
        update_recommended: bool = False
        sample_count: int = 0
        message: str = ""

    # ========================================================================
    # UNIFIED PIPELINE MODELS (v0.11.0)
    # ========================================================================

    class UnifiedSynthesizeRequest(BaseModel):
        """Request for unified pipeline synthesis."""
        text: Optional[str] = Field(None, description="Text to synthesize")
        ssml: Optional[str] = Field(None, description="SSML markup")
        voice_id: Optional[str] = Field(None, description="Voice ID")
        audio_input: Optional[str] = Field(None, description="Base64 audio for speaker ID")

        # Options
        require_biometric_verification: bool = Field(False, description="Require biometric verification")
        enable_quality_monitoring: bool = Field(True, description="Enable quality monitoring")
        stream: bool = Field(False, description="Stream output")

        # Emotion/prosody
        emotion_preset: Optional[str] = Field(None, description="Emotion preset")
        speaking_rate: float = Field(1.0, ge=0.5, le=2.0, description="Speaking rate")

        # Context
        context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
        user_id: Optional[str] = Field(None, description="User ID")
        session_id: Optional[str] = Field(None, description="Session ID")

    class UnifiedStageResult(BaseModel):
        """Result from a pipeline stage."""
        stage: str
        success: bool
        duration_ms: float = 0.0
        message: str = ""
        warnings: List[str] = []
        error: Optional[str] = None

    class UnifiedSynthesizeResponse(BaseModel):
        """Response from unified pipeline."""
        request_id: str
        status: str
        stages_completed: List[str] = []

        # Key results
        voice_id_used: Optional[str] = None
        governed_text: Optional[str] = None
        audio_base64: Optional[str] = None
        audio_duration_seconds: float = 0.0

        # Quality
        quality_score: Optional[float] = None
        quality_gate: Optional[str] = None

        # Metadata
        total_duration_ms: float = 0.0
        error: Optional[str] = None
        warnings: List[str] = []

    # ========================================================================
    # RESOURCE GOVERNANCE MODELS (v0.12.0)
    # ========================================================================

    class RateLimitStatusResponse(BaseModel):
        """Rate limit status response."""
        allowed: bool
        remaining: int
        limit: int
        reset_at: float
        retry_after: Optional[float] = None
        tier: str = "free"

    class QuotaStatusResponse(BaseModel):
        """Quota status response."""
        quota_name: str
        period: str
        limit: int
        used: int
        remaining: int
        resets_at: float
        exceeded: bool = False
        warning_threshold_reached: bool = False

    class QuotaStatusListResponse(BaseModel):
        """Response listing all quota statuses."""
        user_id: str
        tier: str
        quotas: Dict[str, Any]

    class PolicyCheckRequest(BaseModel):
        """Request to check content/usage policy."""
        text: str = Field(..., description="Text to check")
        operation: str = Field("synthesize", description="Operation type")
        voice_id: Optional[str] = Field(None, description="Voice ID")
        is_commercial: bool = Field(False, description="Is commercial use")

    class PolicyCheckResponse(BaseModel):
        """Response from policy check."""
        allowed: bool
        action: str
        violations: List[str] = []
        warnings: List[str] = []
        policies_checked: List[str] = []

    class ResourceStatusResponse(BaseModel):
        """Response showing resource status."""
        resources: Dict[str, Any]
        queue: Dict[str, Any]

    class GovernanceConfigResponse(BaseModel):
        """Response showing current governance config."""
        rate_limits: Dict[str, Any]
        quotas: Dict[str, Any]
        policies: Dict[str, Any]
        security: Dict[str, Any]


# ============================================================================
# API DEPENDENCIES
# ============================================================================

class AuditLogger:
    """Simple audit logger for governance decisions."""

    def __init__(self, log_dir: str = "/tmp/axiom_vox_audit"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log(
        self,
        request_id: str,
        request_type: str,
        request_data: Dict[str, Any],
        result: Dict[str, Any],
        api_key_hash: Optional[str] = None,
    ):
        """Log a governance decision."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "request_type": request_type,
            "api_key_hash": api_key_hash,
            "request": request_data,
            "result": result,
        }

        log_file = os.path.join(
            self.log_dir,
            f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )

        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[datetime]] = {}

    def check(self, key: str) -> bool:
        """Check if request is allowed."""
        now = datetime.now()
        minute_ago = datetime.now().replace(second=0, microsecond=0)

        if key not in self.requests:
            self.requests[key] = []

        # Clean old requests
        self.requests[key] = [
            t for t in self.requests[key]
            if t > minute_ago
        ]

        if len(self.requests[key]) >= self.requests_per_minute:
            return False

        self.requests[key].append(now)
        return True


# ============================================================================
# API FACTORY
# ============================================================================

def create_governed_tts_app(
    title: str = "AXIØM VØX Governed TTS API",
    version: str = "0.1.0",
    require_api_key: bool = True,
    api_keys: Optional[List[str]] = None,
    rate_limit: int = 60,
    strict_governance: bool = False,
    enable_cloning: bool = True,
    audit_logging: bool = True,
) -> "FastAPI":
    """
    Create a governed TTS API application.

    Args:
        title: API title
        version: API version
        require_api_key: Whether to require API key authentication
        api_keys: List of valid API keys (if None, uses AXIOM_VOX_API_KEYS env)
        rate_limit: Requests per minute per key
        strict_governance: If True, refuse any governance violations
        enable_cloning: Whether to enable voice cloning endpoint
        audit_logging: Whether to log all governance decisions

    Returns:
        FastAPI application
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")

    app = FastAPI(
        title=title,
        version=version,
        description="Governed text-to-speech API with AXIØM ethics enforcement",
    )

    # Initialize components
    governor = VoxGovernor(strict_mode=strict_governance)
    voice_boundaries = VoiceBoundaries()
    prosody_guardrails = ProsodyGuardrails()
    rate_limiter = RateLimiter(rate_limit)
    audit_logger = AuditLogger() if audit_logging else None

    # API keys
    valid_keys = set(api_keys or [])
    if not valid_keys:
        env_keys = os.environ.get("AXIOM_VOX_API_KEYS", "")
        if env_keys:
            valid_keys = set(env_keys.split(","))

    # ========================================================================
    # DEPENDENCIES
    # ========================================================================

    async def verify_api_key(x_api_key: str = Header(None)) -> str:
        """Verify API key."""
        if not require_api_key:
            return "anonymous"

        if not x_api_key:
            raise HTTPException(status_code=401, detail="API key required")

        if valid_keys and x_api_key not in valid_keys:
            raise HTTPException(status_code=403, detail="Invalid API key")

        # Rate limit by key
        key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()[:16]
        if not rate_limiter.check(key_hash):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        return key_hash

    # ========================================================================
    # ENDPOINTS
    # ========================================================================

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "axiom-vox-governed-tts",
            "version": version,
            "governance_enabled": True,
            "strict_mode": strict_governance,
        }

    @app.get("/emotion-presets")
    async def list_emotion_presets():
        """
        List available emotion presets.

        Returns all named emotion configurations with their parameters.
        Use a preset name in the `emotion_preset` field of synthesis requests.
        """
        try:
            from axiom_vox.emotion_presets import list_emotion_presets as get_presets
            return get_presets()
        except ImportError:
            return {
                "neutral": {"valence": 0.0, "arousal": 0.4, "dominance": 0.5, "warmth": 0.5, "confidence": 0.5},
                "joy": {"valence": 0.8, "arousal": 0.7, "dominance": 0.5, "warmth": 0.8, "confidence": 0.7},
                "sadness": {"valence": -0.7, "arousal": 0.2, "dominance": 0.3, "warmth": 0.4, "confidence": 0.3},
                "calm": {"valence": 0.3, "arousal": 0.2, "dominance": 0.5, "warmth": 0.7, "confidence": 0.6},
            }

    @app.post("/check", response_model=CheckResponse)
    async def check_governance(
        request: CheckRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Pre-flight governance check.

        Check if text would pass governance before synthesizing.
        """
        # Build emotional intent if provided
        intent = None
        if request.emotion:
            try:
                intent = EmotionalIntent.from_emotion(EmotionCategory(request.emotion))
            except ValueError:
                intent = EmotionalIntent(target_emotion=request.emotion)

        # Run governance
        result = governor.govern(
            text=request.text,
            voice_id=request.voice_id,
            context=request.context,
            emotional_intent=intent,
        )

        # Audit log
        if audit_logger:
            audit_logger.log(
                request_id=hashlib.sha256(request.text.encode()).hexdigest()[:16],
                request_type="check",
                request_data=request.dict(),
                result=result.to_dict(),
                api_key_hash=api_key,
            )

        return CheckResponse(
            would_pass=result.action != GovernanceAction.REFUSE,
            action=result.action.value,
            reason=result.refusal_reason or "Governance passed",
            repairs_needed=result.repairs_made,
            warnings=result.warnings,
        )

    @app.post("/synthesize", response_model=SynthesizeResponse)
    async def synthesize(
        request: SynthesizeRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Synthesize speech with AXIØM governance.

        Supports:
        - Plain text input (text field)
        - SSML markup input (ssml field)
        - Emotion presets (emotion_preset field)
        - Fine-grained emotion control (valence, arousal, etc.)

        Text/SSML is governed before synthesis. Refused content returns an error.
        """
        # Validate input: need text or ssml
        if not request.text and not request.ssml:
            raise HTTPException(
                status_code=400,
                detail="Either 'text' or 'ssml' must be provided"
            )

        # Handle SSML input
        ssml_doc = None
        text_for_governance = request.text

        if request.ssml:
            try:
                from axiom_vox.ssml import SSMLParser
                parser = SSMLParser()
                ssml_doc, ssml_warnings = parser.parse(request.ssml)
                text_for_governance = ssml_doc.plain_text

                # Validate SSML against prosody guardrails
                ssml_decision = prosody_guardrails.validate_ssml(ssml_doc, request.context)
                if not ssml_decision.approved:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "ssml_validation_failed",
                            "reason": ssml_decision.reason,
                            "warnings": ssml_warnings,
                            "detected_patterns": [p.value for p in ssml_decision.detected_patterns],
                        }
                    )
            except ImportError:
                # SSML module not available, treat ssml as plain text
                text_for_governance = request.ssml
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"SSML parsing error: {str(e)}"
                )

        # Build emotional intent from preset or explicit values
        intent = None

        # Start with preset if specified
        if request.emotion_preset:
            try:
                from axiom_vox.emotion_presets import create_intent_from_preset
                intent = create_intent_from_preset(request.emotion_preset)
            except (ImportError, KeyError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid emotion preset: {request.emotion_preset}"
                )

        # Fall back to legacy emotion field
        elif request.emotion:
            try:
                intent = EmotionalIntent.from_emotion(EmotionCategory(request.emotion))
            except ValueError:
                intent = EmotionalIntent(target_emotion=request.emotion)

        # If any explicit values provided, override preset values
        if any(v is not None for v in [request.valence, request.arousal, request.dominance, request.warmth, request.confidence]):
            intent = intent or EmotionalIntent()
            if request.valence is not None:
                intent.valence = request.valence
            if request.arousal is not None:
                intent.arousal = request.arousal
            if request.dominance is not None:
                intent.dominance = request.dominance
            if request.warmth is not None:
                intent.warmth = request.warmth
            if request.confidence is not None:
                intent.confidence = request.confidence

        # Apply speaking rate
        if intent and request.speaking_rate:
            intent.speaking_rate = request.speaking_rate

        # Run governance
        result = governor.govern(
            text=text_for_governance,
            voice_id=request.voice_id,
            context=request.context,
            emotional_intent=intent,
            user_prompt=request.user_prompt,
        )

        # Audit log
        request_id = hashlib.sha256(
            f"{request.text}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        if audit_logger:
            background_tasks.add_task(
                audit_logger.log,
                request_id=request_id,
                request_type="synthesize",
                request_data=request.dict(),
                result=result.to_dict(),
                api_key_hash=api_key,
            )

        # Handle refusal
        if result.action == GovernanceAction.REFUSE:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "governance_refused",
                    "reason": result.refusal_reason,
                    "governance_report": result.to_dict(),
                }
            )

        # === SYNTHESIS WOULD HAPPEN HERE ===
        # In production, this would call the actual VØX TTS engine:
        #
        # audio = await vox_tts.synthesize(
        #     text=result.governed_text,
        #     voice_id=result.voice_id,
        #     emotion=intent,
        #     format=request.output_format,
        # )
        #
        # For now, return a placeholder response

        return SynthesizeResponse(
            success=True,
            action=result.action.value,
            governed_text=result.governed_text,
            audio_url=f"/audio/{request_id}.{request.output_format}",  # Placeholder
            audio_base64=None,  # Would contain actual audio
            duration_seconds=None,  # Would be calculated
            governance_report=result.to_dict(),
            warnings=result.warnings,
        )

    @app.get("/voices", response_model=List[VoiceInfo])
    async def list_voices(
        api_key: str = Depends(verify_api_key),
    ):
        """List available voices."""
        # In production, this would query the voice database
        # For now, return built-in voices
        voices = [
            VoiceInfo(
                voice_id="axiom_default",
                name="AXIØM Default",
                category="synthetic",
                description="Balanced, professional voice",
                languages=["en"],
            ),
            VoiceInfo(
                voice_id="axiom_warm",
                name="AXIØM Warm",
                category="synthetic",
                description="Friendly, approachable voice",
                languages=["en"],
            ),
            VoiceInfo(
                voice_id="axiom_professional",
                name="AXIØM Professional",
                category="synthetic",
                description="Authoritative, clear voice",
                languages=["en"],
            ),
            VoiceInfo(
                voice_id="alloy",
                name="Alloy",
                category="synthetic",
                description="Neutral synthetic voice",
                languages=["en"],
            ),
            VoiceInfo(
                voice_id="nova",
                name="Nova",
                category="synthetic",
                description="Expressive synthetic voice",
                languages=["en"],
            ),
        ]
        return voices

    @app.get("/voice/{voice_id}", response_model=VoiceInfo)
    async def get_voice(
        voice_id: str,
        api_key: str = Depends(verify_api_key),
    ):
        """Get details about a specific voice."""
        # Check voice boundaries for clearance status
        request = VoiceCloneRequest(
            voice_id=voice_id,
            intended_use="general",
            content_preview="Test content for voice check.",
        )
        decision = voice_boundaries.check(request)

        return VoiceInfo(
            voice_id=voice_id,
            name=voice_id.replace("_", " ").title(),
            category=decision.voice_category.value,
            description=f"Voice cleared: {decision.approved}",
            languages=["en"],
            is_cloned=decision.voice_category != VoiceCategory.SYNTHETIC,
            consent_verified=decision.approved,
            allowed_uses=["general"] if decision.approved else [],
        )

    if enable_cloning:
        @app.post("/clone", response_model=CloneResponse)
        async def clone_voice(
            request: CloneRequest,
            background_tasks: BackgroundTasks,
            api_key: str = Depends(verify_api_key),
        ):
            """
            Request to clone a voice.

            Subject to ethics review and consent verification.
            """
            # Generate voice ID
            voice_id = f"clone_{hashlib.sha256(request.voice_name.encode()).hexdigest()[:12]}"

            # Check ethics
            clone_request = VoiceCloneRequest(
                voice_id=voice_id,
                intended_use=request.intended_use,
                content_preview=f"Clone request for: {request.voice_name}",
                has_consent_proof=request.consent_proof is not None,
                requestor_id=request.owner_id,
            )

            decision = voice_boundaries.check(clone_request)

            # Audit log
            if audit_logger:
                background_tasks.add_task(
                    audit_logger.log,
                    request_id=voice_id,
                    request_type="clone",
                    request_data={
                        "voice_name": request.voice_name,
                        "intended_use": request.intended_use,
                        "has_consent_proof": request.consent_proof is not None,
                    },
                    result=decision.to_dict(),
                    api_key_hash=api_key,
                )

            if not decision.approved:
                return CloneResponse(
                    approved=False,
                    voice_id=None,
                    reason=decision.reason,
                    required_disclaimers=decision.required_disclaimers,
                    usage_restrictions=decision.usage_restrictions,
                )

            # === ACTUAL CLONING WOULD HAPPEN HERE ===
            # In production:
            # voice_id = await vox_cloner.clone(request.audio_samples)
            # voice_boundaries.register_voice(voice_id, ...)

            return CloneResponse(
                approved=True,
                voice_id=voice_id,
                reason="Clone request approved - processing",
                required_disclaimers=decision.required_disclaimers,
                usage_restrictions=decision.usage_restrictions,
            )

    # ========================================================================
    # FINE-TUNING ENDPOINTS
    # ========================================================================

    @app.post("/finetune", response_model=FineTuneResponse)
    async def start_finetune(
        request: FineTuneRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Start a voice fine-tuning job.

        Creates a LoRA adapter trained on the provided audio samples.
        Requires consent verification for production use.
        """
        manager = _get_job_manager()
        if manager is None:
            raise HTTPException(
                status_code=503,
                detail="Fine-tuning service not available"
            )

        # Generate voice ID
        voice_id = f"clone_{hashlib.sha256(request.voice_name.encode()).hexdigest()[:12]}"

        # Check ethics via voice boundaries
        clone_request = VoiceCloneRequest(
            voice_id=voice_id,
            intended_use=request.intended_use,
            content_preview=f"Fine-tune request for: {request.voice_name}",
            has_consent_proof=request.consent_verified,
            requestor_id=request.owner_id,
        )

        decision = voice_boundaries.check(clone_request)

        if not decision.approved:
            return FineTuneResponse(
                success=False,
                job_id=None,
                voice_id=None,
                status="rejected",
                message=decision.reason,
            )

        try:
            # Create the job
            job_id = await manager.create_job(
                voice_id=voice_id,
                audio_files=request.audio_files,
                consent_verified=request.consent_verified,
                requestor_id=request.owner_id,
                epochs=request.epochs,
                fast_mode=request.fast_mode,
            )

            # Start the job in background
            background_tasks.add_task(
                manager.start_job,
                job_id,
                request.audio_files,
                {"fast": request.fast_mode},
            )

            # Audit log
            if audit_logger:
                background_tasks.add_task(
                    audit_logger.log,
                    request_id=job_id,
                    request_type="finetune_start",
                    request_data={
                        "voice_name": request.voice_name,
                        "voice_id": voice_id,
                        "sample_count": len(request.audio_files),
                        "consent_verified": request.consent_verified,
                        "epochs": request.epochs,
                        "fast_mode": request.fast_mode,
                    },
                    result={"job_id": job_id, "status": "pending"},
                    api_key_hash=api_key,
                )

            return FineTuneResponse(
                success=True,
                job_id=job_id,
                voice_id=voice_id,
                status="pending",
                message="Fine-tuning job created and queued",
                estimated_duration_minutes=15.0 if request.fast_mode else 30.0,
            )

        except Exception as e:
            logger.exception(f"Failed to create fine-tuning job: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create job: {str(e)}"
            )

    @app.get("/finetune/{job_id}", response_model=JobStatusResponse)
    async def get_finetune_status(
        job_id: str,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Get the status of a fine-tuning job.

        Returns progress, metrics, and completion status.
        """
        manager = _get_job_manager()
        if manager is None:
            raise HTTPException(
                status_code=503,
                detail="Fine-tuning service not available"
            )

        job = await manager.get_status(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )

        return JobStatusResponse(
            job_id=job.job_id,
            voice_id=job.voice_id,
            status=job.status.value,
            progress=job.progress,
            current_epoch=job.current_epoch,
            total_epochs=job.total_epochs,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            estimated_remaining_seconds=job.estimated_remaining_seconds,
            final_loss=job.final_loss,
            similarity_score=job.similarity_score,
            quality_score=job.quality_score,
            verification_passed=job.verification_passed,
            error_message=job.error_message,
        )

    @app.delete("/finetune/{job_id}", response_model=JobCancelResponse)
    async def cancel_finetune(
        job_id: str,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Cancel a fine-tuning job.

        Can cancel pending or in-progress jobs.
        """
        manager = _get_job_manager()
        if manager is None:
            raise HTTPException(
                status_code=503,
                detail="Fine-tuning service not available"
            )

        job = await manager.get_status(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )

        try:
            cancelled = await manager.cancel_job(job_id)

            # Audit log
            if audit_logger:
                background_tasks.add_task(
                    audit_logger.log,
                    request_id=job_id,
                    request_type="finetune_cancel",
                    request_data={"job_id": job_id},
                    result={"cancelled": cancelled},
                    api_key_hash=api_key,
                )

            if cancelled:
                return JobCancelResponse(
                    success=True,
                    job_id=job_id,
                    message="Job cancelled successfully",
                )
            else:
                return JobCancelResponse(
                    success=False,
                    job_id=job_id,
                    message=f"Cannot cancel job in status: {job.status.value}",
                )

        except Exception as e:
            logger.exception(f"Failed to cancel job {job_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to cancel job: {str(e)}"
            )

    @app.get("/finetune", response_model=List[JobStatusResponse])
    async def list_finetune_jobs(
        voice_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        api_key: str = Depends(verify_api_key),
    ):
        """
        List fine-tuning jobs.

        Optionally filter by voice_id or status.
        """
        manager = _get_job_manager()
        if manager is None:
            raise HTTPException(
                status_code=503,
                detail="Fine-tuning service not available"
            )

        try:
            from axiom_vox.finetuning import JobStatus as FTJobStatus

            status_filter = None
            if status:
                try:
                    status_filter = FTJobStatus(status)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid status: {status}"
                    )

            jobs = await manager.list_jobs(
                voice_id=voice_id,
                status=status_filter,
                limit=limit,
            )

            return [
                JobStatusResponse(
                    job_id=job.job_id,
                    voice_id=job.voice_id,
                    status=job.status.value,
                    progress=job.progress,
                    current_epoch=job.current_epoch,
                    total_epochs=job.total_epochs,
                    created_at=job.created_at.isoformat(),
                    started_at=job.started_at.isoformat() if job.started_at else None,
                    completed_at=job.completed_at.isoformat() if job.completed_at else None,
                    estimated_remaining_seconds=job.estimated_remaining_seconds,
                    final_loss=job.final_loss,
                    similarity_score=job.similarity_score,
                    quality_score=job.quality_score,
                    verification_passed=job.verification_passed,
                    error_message=job.error_message,
                )
                for job in jobs
            ]

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Failed to list jobs: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list jobs: {str(e)}"
            )

    # ========================================================================
    # MULTI-VOICE ENDPOINTS (v0.9.0)
    # ========================================================================

    @app.post("/synthesize/multi-voice", response_model=MultiVoiceSynthesizeResponse)
    async def synthesize_multi_voice(
        request: MultiVoiceSynthesizeRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Synthesize multi-voice audio from SSML or dialogue script.

        Enables seamless synthesis across multiple voices with automatic
        transitions and character-to-voice mapping.

        Input formats (provide one):
        - ssml: SSML markup with <voice> tags
        - script: Structured DialogueScript with lines

        Example SSML:
            <speak>
                <voice axiom-voice="professional">Welcome!</voice>
                <voice axiom-voice="expert" emotion="confident">Let me explain.</voice>
            </speak>

        Example script:
            {
                "lines": [
                    {"text": "Welcome!", "voice_id": "professional"},
                    {"text": "Let me explain.", "voice_id": "expert", "emotion": "confident"}
                ],
                "default_transition": "breath_pause"
            }
        """
        import base64

        try:
            from axiom_vox.multi_voice_synthesizer import get_multi_voice_synthesizer
            from axiom_vox.multi_voice import DialogueScript, DialogueLine, TransitionStyle
        except ImportError as e:
            logger.error(f"Multi-voice module not available: {e}")
            raise HTTPException(
                status_code=503,
                detail="Multi-voice synthesis not available"
            )

        # Validate input
        if not request.ssml and not request.script:
            raise HTTPException(
                status_code=400,
                detail="Either 'ssml' or 'script' must be provided"
            )

        synthesizer = get_multi_voice_synthesizer()

        try:
            if request.ssml:
                # Synthesize from SSML
                result = synthesizer.synthesize_ssml(
                    ssml=request.ssml,
                    default_voice_id=request.default_voice_id,
                    output_format=request.output_format,
                )
            else:
                # Synthesize from script
                lines = [
                    DialogueLine(
                        text=line.text,
                        voice_id=line.voice_id,
                        character_name=line.character_name,
                        emotion=line.emotion,
                        pause_before_ms=line.pause_before_ms,
                        pause_after_ms=line.pause_after_ms,
                    )
                    for line in request.script.lines
                ]

                try:
                    transition = TransitionStyle(request.script.default_transition)
                except ValueError:
                    transition = TransitionStyle.BREATH_PAUSE

                script = DialogueScript(
                    lines=lines,
                    default_transition=transition,
                )

                result = synthesizer.synthesize_script(
                    script=script,
                    output_format=request.output_format,
                )

            # Audit log
            if audit_logger:
                background_tasks.add_task(
                    audit_logger.log,
                    request_id=f"multi_voice_{int(datetime.now().timestamp()*1000)}",
                    request_type="synthesize_multi_voice",
                    request_data={
                        "has_ssml": request.ssml is not None,
                        "script_lines": len(request.script.lines) if request.script else 0,
                        "voices_used": result.voices_used,
                    },
                    result={"success": result.success, "errors": result.errors},
                    api_key_hash=api_key,
                )

            # Convert segments to response format
            segments = [
                SegmentInfo(
                    text=seg.text,
                    voice_id=seg.voice_id,
                    character_name=seg.character_name,
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    duration_ms=seg.end_ms - seg.start_ms,
                )
                for seg in result.segments
            ]

            # Convert voice switches to response format
            voice_switches = [
                VoiceSwitchInfo(
                    from_voice=sw.from_voice,
                    to_voice=sw.to_voice,
                    transition_style=sw.transition_style.value if hasattr(sw.transition_style, 'value') else str(sw.transition_style),
                    word_index=sw.word_index,
                    character_from=sw.character_from,
                    character_to=sw.character_to,
                )
                for sw in result.voice_switches
            ]

            # Encode audio as base64
            audio_base64 = None
            if result.audio_bytes:
                audio_base64 = base64.b64encode(result.audio_bytes).decode('utf-8')

            return MultiVoiceSynthesizeResponse(
                success=result.success,
                total_duration_ms=result.total_duration_ms,
                synthesis_time_ms=result.synthesis_time_ms,
                segments=segments,
                voice_switches=voice_switches,
                voices_used=result.voices_used,
                audio_base64=audio_base64,
                errors=result.errors,
            )

        except Exception as e:
            logger.exception(f"Multi-voice synthesis failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Synthesis failed: {str(e)}"
            )

    @app.post("/synthesize/multi-voice/stream")
    async def synthesize_multi_voice_stream(
        request: MultiVoiceSynthesizeRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Stream multi-voice synthesis with voice switch events.

        Returns audio chunks interspersed with voice switch events.

        Response format (binary stream):
        - [4 bytes: JSON length][JSON metadata]
        - [4 bytes: event length | 0x80000000][voice_switch event]
        - [4 bytes: chunk length][audio chunk]
        - ...
        - [4 bytes: 0x00000000] (end marker)
        """
        try:
            from axiom_vox.multi_voice_synthesizer import get_multi_voice_synthesizer
            from axiom_vox.multi_voice import DialogueScript, DialogueLine, TransitionStyle
        except ImportError as e:
            logger.error(f"Multi-voice module not available: {e}")
            raise HTTPException(
                status_code=503,
                detail="Multi-voice streaming not available"
            )

        # Validate input
        if not request.ssml and not request.script:
            raise HTTPException(
                status_code=400,
                detail="Either 'ssml' or 'script' must be provided"
            )

        synthesizer = get_multi_voice_synthesizer()

        # Build script from input
        if request.ssml:
            from axiom_vox.ssml import SSMLParser
            parser = SSMLParser()
            doc, _ = parser.parse(request.ssml)

            lines = []
            for voice_span in doc.voice_spans:
                voice_id = voice_span.get_resolved_voice_id() or request.default_voice_id
                lines.append(DialogueLine(
                    text=voice_span.text,
                    voice_id=voice_id,
                    character_name=voice_span.character,
                    emotion=voice_span.emotion,
                ))

            script = DialogueScript(lines=lines)
        else:
            lines = [
                DialogueLine(
                    text=line.text,
                    voice_id=line.voice_id,
                    character_name=line.character_name,
                    emotion=line.emotion,
                    pause_before_ms=line.pause_before_ms,
                    pause_after_ms=line.pause_after_ms,
                )
                for line in request.script.lines
            ]

            try:
                transition = TransitionStyle(request.script.default_transition)
            except ValueError:
                transition = TransitionStyle.BREATH_PAUSE

            script = DialogueScript(lines=lines, default_transition=transition)

        async def generate():
            """Generator for StreamingResponse."""
            # Send metadata first
            metadata = {
                "status": "started",
                "total_lines": len(script.lines),
                "voices": list(script.voices_used),
                "sample_rate": 24000,
                "format": request.output_format,
            }
            metadata_json = json.dumps(metadata).encode('utf-8')
            yield struct.pack('>I', len(metadata_json)) + metadata_json

            # Stream synthesis
            async for msg_type, data in synthesizer.synthesize_script_stream(script):
                if msg_type == "voice_switch":
                    # Send voice switch event
                    event = {"type": "voice_switch", **data}
                    event_json = json.dumps(event).encode('utf-8')
                    yield struct.pack('>I', len(event_json) | 0x80000000) + event_json

                elif msg_type == "segment":
                    # Send segment info event
                    event = {"type": "segment", **data}
                    event_json = json.dumps(event).encode('utf-8')
                    yield struct.pack('>I', len(event_json) | 0x80000000) + event_json

                elif msg_type == "audio":
                    # Send audio chunk
                    yield struct.pack('>I', len(data)) + data

                elif msg_type == "error":
                    # Send error event
                    event = {"type": "error", **data}
                    event_json = json.dumps(event).encode('utf-8')
                    yield struct.pack('>I', len(event_json) | 0x80000000) + event_json

                elif msg_type == "completed":
                    # Send completion event
                    event = {"type": "completed", **data}
                    event_json = json.dumps(event).encode('utf-8')
                    yield struct.pack('>I', len(event_json) | 0x80000000) + event_json

            # End marker
            yield struct.pack('>I', 0)

        return StreamingResponse(
            generate(),
            media_type="application/octet-stream",
            headers={
                "X-Multi-Voice": "true",
                "X-Audio-Sample-Rate": "24000",
                "X-Audio-Format": request.output_format,
            }
        )

    # ========================================================================
    # STREAMING ENDPOINTS
    # ========================================================================

    @app.post("/synthesize/stream")
    async def synthesize_stream(
        request: StreamSynthesizeRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Stream synthesized audio with AXIOM governance.

        Returns audio chunks as they're generated via StreamingResponse.
        Governance check happens before streaming starts.

        Response format (binary stream):
        - [4 bytes: JSON length][JSON metadata]
        - [4 bytes: chunk length][audio chunk 1]
        - [4 bytes: chunk length][audio chunk 2]
        - ...
        - [4 bytes: 0x00000000] (end marker)

        High bit in length = metadata (timing, sentence boundaries)
        """
        try:
            from axiom_vox.streaming import get_stream_manager, MessageType
            from axiom_vox.synthesis import VoiceConfig
        except ImportError as e:
            logger.error(f"Streaming module not available: {e}")
            raise HTTPException(
                status_code=503,
                detail="Streaming service not available"
            )

        stream_manager = get_stream_manager()

        # Create session
        session = stream_manager.create_session(
            text=request.text,
            voice_id=request.voice_id,
        )

        # Build emotional intent
        intent = None
        if request.emotion or request.arousal is not None:
            intent = EmotionalIntent(
                target_emotion=request.emotion,
                arousal=request.arousal or 0.5,
                valence=request.valence or 0.0,
                speaking_rate=request.speaking_rate or 1.0,
            )

        # Pre-flight governance check
        if not await stream_manager.run_governance(session, request.context, intent):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "governance_refused",
                    "reason": session.error_message,
                    "governance_report": session.governance_report,
                }
            )

        # Load voice adapter if needed
        if not await stream_manager.load_voice(session):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "voice_load_failed",
                    "reason": session.error_message,
                }
            )

        # Audit log
        if audit_logger:
            background_tasks.add_task(
                audit_logger.log,
                request_id=session.request_id,
                request_type="synthesize_stream",
                request_data=request.dict(),
                result={"action": "streaming", "governance": session.governance_report},
                api_key_hash=api_key,
            )

        async def generate():
            """Generator for StreamingResponse."""
            # Send metadata first
            metadata = {
                "request_id": session.request_id,
                "status": "started",
                "total_sentences": session.total_sentences,
                "governance_action": session.governance_report.get("action") if session.governance_report else "allow",
                "sample_rate": 24000,
                "format": "wav",
            }
            metadata_json = json.dumps(metadata).encode('utf-8')
            yield struct.pack('>I', len(metadata_json)) + metadata_json

            # Build voice config
            voice_config = VoiceConfig(
                voice_id=request.voice_id,
                speaking_rate=request.speaking_rate or 1.0,
                emotion=request.emotion,
            )

            # Stream audio chunks
            async for message in stream_manager.stream(session, voice_config):
                if message.type == MessageType.CHUNK and message.chunk:
                    # Send chunk with length prefix
                    chunk_data = message.chunk.data
                    yield struct.pack('>I', len(chunk_data)) + chunk_data

                    # Optionally include timing metadata
                    if request.include_timings and message.chunk.is_sentence_end:
                        timing = {
                            "type": "sentence_boundary",
                            "index": message.chunk.sentence_index,
                            "timestamp_ms": message.chunk.timestamp_ms,
                        }
                        timing_json = json.dumps(timing).encode('utf-8')
                        # High bit indicates metadata
                        yield struct.pack('>I', len(timing_json) | 0x80000000) + timing_json

                elif message.type == MessageType.ERROR:
                    error = {"type": "error", "error": message.error}
                    error_json = json.dumps(error).encode('utf-8')
                    yield struct.pack('>I', len(error_json) | 0x80000000) + error_json
                    break

            # End marker
            yield struct.pack('>I', 0)

            # Cleanup session
            stream_manager.remove_session(session.request_id)

        return StreamingResponse(
            generate(),
            media_type="application/octet-stream",
            headers={
                "X-Request-ID": session.request_id,
                "X-Audio-Sample-Rate": "24000",
                "X-Audio-Format": "wav",
            }
        )

    @app.websocket("/ws/synthesize")
    async def websocket_synthesize(websocket: WebSocket):
        """
        WebSocket endpoint for interactive streaming TTS.

        Protocol:

        Client -> Server:
            {"type": "synthesize", "text": "...", "voice_id": "...", ...}
            {"type": "pause"}
            {"type": "resume"}
            {"type": "cancel"}

        Server -> Client:
            {"type": "started", "request_id": "...", "total_sentences": 3, ...}
            {"type": "governance_passed", "governance_report": {...}}
            {"type": "chunk", "chunk": {"index": 0, "size_bytes": 4096, ...}}
            // Binary frame with actual audio data follows each chunk message
            {"type": "sentence_boundary", "sentence_index": 0}
            {"type": "progress", "progress": {...}}
            {"type": "completed", "metadata": {...}}
            {"type": "error", "error": "..."}
        """
        await websocket.accept()

        try:
            from axiom_vox.streaming import get_stream_manager, StreamSession, MessageType
            from axiom_vox.synthesis import VoiceConfig
        except ImportError as e:
            await websocket.send_json({"type": "error", "error": f"Streaming not available: {e}"})
            await websocket.close()
            return

        stream_manager = get_stream_manager()
        current_session: Optional[StreamSession] = None

        try:
            while True:
                # Receive message from client
                try:
                    data = await websocket.receive_json()
                except Exception:
                    # Might be a close frame
                    break

                msg_type = data.get("type", "")

                # Handle control messages
                if msg_type == "pause" and current_session:
                    current_session.state = StreamSession.State.PAUSED
                    await websocket.send_json({"type": "paused", "request_id": current_session.request_id})
                    continue

                elif msg_type == "resume" and current_session:
                    if current_session.state == StreamSession.State.PAUSED:
                        current_session.state = StreamSession.State.STREAMING
                    await websocket.send_json({"type": "resumed", "request_id": current_session.request_id})
                    continue

                elif msg_type == "cancel" and current_session:
                    stream_manager.cancel_session(current_session.request_id)
                    await websocket.send_json({"type": "cancelled", "request_id": current_session.request_id})
                    current_session = None
                    continue

                elif msg_type != "synthesize":
                    await websocket.send_json({"type": "error", "error": f"Unknown message type: {msg_type}"})
                    continue

                # Handle synthesis request
                text = data.get("text", "")
                voice_id = data.get("voice_id", "axiom_default")
                emotion = data.get("emotion")
                context = data.get("context", {})

                if not text:
                    await websocket.send_json({"type": "error", "error": "No text provided"})
                    continue

                # Create session
                current_session = stream_manager.create_session(text, voice_id)

                # Build emotional intent
                intent = None
                if emotion:
                    intent = EmotionalIntent(
                        target_emotion=emotion,
                        arousal=data.get("arousal", 0.5),
                        valence=data.get("valence", 0.0),
                        speaking_rate=data.get("speaking_rate", 1.0),
                    )

                # Pre-flight governance
                if not await stream_manager.run_governance(current_session, context, intent):
                    await websocket.send_json({
                        "type": "governance_refused",
                        "request_id": current_session.request_id,
                        "error": current_session.error_message,
                        "governance_report": current_session.governance_report,
                    })
                    stream_manager.remove_session(current_session.request_id)
                    current_session = None
                    continue

                # Load voice
                if not await stream_manager.load_voice(current_session):
                    await websocket.send_json({
                        "type": "error",
                        "request_id": current_session.request_id,
                        "error": current_session.error_message,
                    })
                    stream_manager.remove_session(current_session.request_id)
                    current_session = None
                    continue

                # Build voice config
                voice_config = VoiceConfig(
                    voice_id=voice_id,
                    speaking_rate=data.get("speaking_rate", 1.0),
                    emotion=emotion,
                )

                # Stream synthesis
                async for message in stream_manager.stream(current_session, voice_config):
                    # Check for pause
                    while current_session.state == StreamSession.State.PAUSED:
                        await asyncio.sleep(0.1)
                        # Check for cancel during pause
                        if current_session.state == StreamSession.State.CANCELLED:
                            break

                    if current_session.state == StreamSession.State.CANCELLED:
                        break

                    # Send JSON message
                    msg_json = message.to_json()
                    await websocket.send_json(msg_json)

                    # Send binary audio data for chunks
                    if message.type == MessageType.CHUNK and message.chunk:
                        await websocket.send_bytes(message.chunk.data)

                # Cleanup
                stream_manager.remove_session(current_session.request_id)
                current_session = None

        except WebSocketDisconnect:
            if current_session:
                stream_manager.cancel_session(current_session.request_id)
                stream_manager.remove_session(current_session.request_id)
        except Exception as e:
            logger.exception(f"WebSocket error: {e}")
            try:
                await websocket.send_json({"type": "error", "error": str(e)})
            except:
                pass
            if current_session:
                stream_manager.remove_session(current_session.request_id)

    # ========================================================================
    # BIOMETRIC ENDPOINTS (v0.10.0)
    # ========================================================================

    @app.post("/biometrics/enroll", response_model=BiometricEnrollResponse)
    async def biometric_enroll(
        request: BiometricEnrollRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Enroll voice biometrics.

        Requires minimum 3 audio samples (~5s each) and consent token.
        """
        try:
            from axiom_vox.biometrics import (
                VoiceBiometricService,
                EnrollmentStatus,
            )
            import base64

            service = VoiceBiometricService()

            # Decode audio samples
            audio_samples = []
            for sample_b64 in request.audio_samples:
                try:
                    audio_bytes = base64.b64decode(sample_b64)
                    audio_samples.append(audio_bytes)
                except Exception as e:
                    return BiometricEnrollResponse(
                        success=False,
                        status="error",
                        message=f"Invalid audio sample: {str(e)}",
                    )

            result = await service.enroll(
                voice_id=request.voice_id,
                audio_samples=audio_samples,
                owner_id=request.owner_id,
                consent_token=request.consent_token,
                metadata=request.metadata,
            )

            return BiometricEnrollResponse(
                success=result.status == EnrollmentStatus.SUCCESS,
                status=result.status.value,
                template_id=result.template_id,
                voice_id=result.voice_id,
                sample_count=result.sample_count,
                confidence=result.confidence,
                message=result.message,
                warnings=result.warnings,
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Biometrics module not available",
            )
        except Exception as e:
            logger.exception(f"Biometric enrollment error: {e}")
            return BiometricEnrollResponse(
                success=False,
                status="error",
                message=str(e),
            )

    @app.post("/biometrics/verify", response_model=BiometricVerifyResponse)
    async def biometric_verify(
        request: BiometricVerifyRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Verify voice against enrolled biometric template.

        Includes liveness detection to prevent replay/deepfake attacks.
        """
        try:
            from axiom_vox.biometrics import (
                VoiceBiometricService,
                VerificationStatus,
            )
            import base64

            service = VoiceBiometricService()

            # Decode audio
            try:
                audio_bytes = base64.b64decode(request.audio_sample)
            except Exception as e:
                return BiometricVerifyResponse(
                    verified=False,
                    status="error",
                    voice_id=request.voice_id,
                    message=f"Invalid audio: {str(e)}",
                )

            result = await service.verify(
                voice_id=request.voice_id,
                audio_sample=audio_bytes,
                require_liveness=request.require_liveness,
            )

            return BiometricVerifyResponse(
                verified=result.is_verified,
                status=result.status.value,
                voice_id=result.voice_id,
                similarity_score=result.similarity_score,
                threshold=result.threshold,
                liveness_passed=result.liveness_passed,
                liveness_score=result.liveness_score,
                drift_detected=result.drift_detected,
                drift_severity=result.drift_severity.value,
                message=result.message,
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Biometrics module not available",
            )
        except Exception as e:
            logger.exception(f"Biometric verification error: {e}")
            return BiometricVerifyResponse(
                verified=False,
                status="error",
                voice_id=request.voice_id,
                message=str(e),
            )

    @app.get("/biometrics/status/{voice_id}", response_model=BiometricStatusResponse)
    async def biometric_status(
        voice_id: str,
        api_key: str = Depends(verify_api_key),
    ):
        """Get enrollment status for a voice."""
        try:
            from axiom_vox.biometrics import VoiceBiometricService

            service = VoiceBiometricService()
            status = await service.get_status(voice_id)

            return BiometricStatusResponse(
                voice_id=status["voice_id"],
                enrolled=status["enrolled"],
                template_id=status.get("template_id"),
                sample_count=status.get("sample_count", 0),
                confidence=status.get("confidence", 0.0),
                enrolled_at=status.get("enrolled_at"),
                updated_at=status.get("updated_at"),
                verification_stats=status.get("verification_stats"),
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Biometrics module not available",
            )

    @app.post("/biometrics/update/{voice_id}", response_model=BiometricEnrollResponse)
    async def biometric_update(
        voice_id: str,
        request: BiometricEnrollRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """Update biometric template with new samples."""
        try:
            from axiom_vox.biometrics import (
                VoiceBiometricService,
                EnrollmentStatus,
            )
            import base64

            service = VoiceBiometricService()

            # Decode audio samples
            audio_samples = []
            for sample_b64 in request.audio_samples:
                try:
                    audio_bytes = base64.b64decode(sample_b64)
                    audio_samples.append(audio_bytes)
                except Exception:
                    continue

            result = await service.update_template(
                voice_id=voice_id,
                audio_samples=audio_samples,
                consent_token=request.consent_token,
            )

            return BiometricEnrollResponse(
                success=result.status == EnrollmentStatus.SUCCESS,
                status=result.status.value,
                template_id=result.template_id,
                voice_id=result.voice_id,
                sample_count=result.sample_count,
                confidence=result.confidence,
                message=result.message,
                warnings=result.warnings,
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Biometrics module not available",
            )

    @app.delete("/biometrics/{voice_id}")
    async def biometric_revoke(
        voice_id: str,
        reason: str = "",
        api_key: str = Depends(verify_api_key),
    ):
        """Revoke biometric enrollment and delete template."""
        try:
            from axiom_vox.biometrics import VoiceBiometricService

            service = VoiceBiometricService()
            revoked = await service.revoke_template(voice_id, reason)

            return {
                "success": revoked,
                "voice_id": voice_id,
                "message": "Template revoked" if revoked else "No template found",
            }

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Biometrics module not available",
            )

    @app.get("/biometrics/drift/{voice_id}", response_model=BiometricDriftResponse)
    async def biometric_drift(
        voice_id: str,
        api_key: str = Depends(verify_api_key),
    ):
        """Get voice drift analysis."""
        try:
            from axiom_vox.biometrics import VoiceBiometricService

            service = VoiceBiometricService()
            drift = await service.check_drift(voice_id)

            return BiometricDriftResponse(
                voice_id=drift.voice_id,
                severity=drift.severity.value,
                short_term_drift=drift.short_term_drift,
                long_term_drift=drift.long_term_drift,
                trend_direction=drift.trend_direction,
                requires_re_enrollment=drift.requires_re_enrollment,
                update_recommended=drift.update_recommended,
                sample_count=drift.sample_count,
                message=drift.message,
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Biometrics module not available",
            )

    # ========================================================================
    # UNIFIED PIPELINE ENDPOINTS (v0.11.0)
    # ========================================================================

    @app.post("/unified/synthesize", response_model=UnifiedSynthesizeResponse)
    async def unified_synthesize(
        request: UnifiedSynthesizeRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """
        Synthesize using unified pipeline.

        Single entry point for all voice operations with:
        - Biometric identification/verification
        - Unified consent checking
        - AXIOM governance
        - Intelligent voice routing
        - Real-time quality monitoring
        """
        try:
            from axiom_vox.unified import (
                VoxUnifiedPipeline,
                PipelineRequest,
                PipelineStatus,
            )
            import base64

            pipeline = VoxUnifiedPipeline()

            # Build request
            audio_input = None
            if request.audio_input:
                audio_input = base64.b64decode(request.audio_input)

            pipe_request = PipelineRequest(
                text=request.text,
                ssml=request.ssml,
                voice_id=request.voice_id,
                audio_input=audio_input,
                require_biometric_verification=request.require_biometric_verification,
                enable_quality_monitoring=request.enable_quality_monitoring,
                stream=request.stream,
                emotion_preset=request.emotion_preset,
                speaking_rate=request.speaking_rate,
                context=request.context or {},
                user_id=request.user_id,
                session_id=request.session_id,
            )

            response = await pipeline.process(pipe_request)

            return UnifiedSynthesizeResponse(
                request_id=response.request_id,
                status=response.status.value,
                stages_completed=[s.value for s in response.stages_completed],
                voice_id_used=response.voice_id_used,
                governed_text=response.governed_text,
                audio_base64=response.audio_base64,
                audio_duration_seconds=(
                    response.synthesis.duration_seconds if response.synthesis else 0
                ),
                quality_score=(
                    response.quality.overall_score if response.quality else None
                ),
                quality_gate=(
                    response.quality.gate_status.value if response.quality else None
                ),
                total_duration_ms=response.total_duration_ms,
                error=response.error,
                warnings=response.warnings,
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Unified pipeline module not available",
            )
        except Exception as e:
            logger.exception(f"Unified pipeline error: {e}")
            return UnifiedSynthesizeResponse(
                request_id="error",
                status="failed",
                error=str(e),
            )

    @app.post("/unified/synthesize/stream")
    async def unified_synthesize_stream(
        request: UnifiedSynthesizeRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """Stream synthesis using unified pipeline."""
        try:
            from axiom_vox.unified import VoxUnifiedPipeline, PipelineRequest
            import base64

            pipeline = VoxUnifiedPipeline()

            audio_input = None
            if request.audio_input:
                audio_input = base64.b64decode(request.audio_input)

            pipe_request = PipelineRequest(
                text=request.text,
                ssml=request.ssml,
                voice_id=request.voice_id,
                audio_input=audio_input,
                require_biometric_verification=request.require_biometric_verification,
                enable_quality_monitoring=request.enable_quality_monitoring,
                stream=True,
                emotion_preset=request.emotion_preset,
                speaking_rate=request.speaking_rate,
                context=request.context or {},
            )

            async def generate():
                async for chunk in pipeline.process_stream(pipe_request):
                    yield json.dumps(chunk) + "\n"

            return StreamingResponse(
                generate(),
                media_type="application/x-ndjson",
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Unified pipeline module not available",
            )

    @app.get("/unified/consent/{voice_id}")
    async def unified_consent_status(
        voice_id: str,
        api_key: str = Depends(verify_api_key),
    ):
        """Get unified consent status for a voice."""
        try:
            from axiom_vox.unified import get_consent_registry

            registry = get_consent_registry()
            status = registry.get_consent_status(voice_id)

            return status

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Unified pipeline module not available",
            )

    # ========================================================================
    # RESOURCE GOVERNANCE ENDPOINTS (v0.12.0)
    # ========================================================================

    @app.get("/governance/rate-limit/status", response_model=RateLimitStatusResponse)
    async def get_rate_limit_status(
        user_id: Optional[str] = None,
        operation: str = "synthesize",
        api_key: str = Depends(verify_api_key),
    ):
        """Get current rate limit status."""
        try:
            from axiom_vox.governance import get_rate_limiter, TieredRateLimiter

            # Use tiered rate limiter if available
            limiter = TieredRateLimiter()
            user = user_id or "anonymous"

            result = limiter.check(user, operation=operation, cost=0)

            return RateLimitStatusResponse(
                allowed=result.allowed,
                remaining=result.remaining,
                limit=result.limit,
                reset_at=result.reset_at,
                retry_after=result.retry_after,
                tier=limiter.get_user_tier(user),
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Governance module not available",
            )

    @app.get("/governance/quotas", response_model=QuotaStatusListResponse)
    async def get_quota_status(
        user_id: str,
        api_key: str = Depends(verify_api_key),
    ):
        """Get all quota statuses for a user."""
        try:
            from axiom_vox.governance import get_quota_manager

            manager = get_quota_manager()
            statuses = manager.get_all_statuses(user_id)

            return QuotaStatusListResponse(
                user_id=user_id,
                tier=manager.get_user_tier(user_id),
                quotas={
                    name: {
                        "limit": s.limit,
                        "used": s.used,
                        "remaining": s.remaining,
                        "resets_at": s.resets_at,
                        "exceeded": s.exceeded,
                    }
                    for name, s in statuses.items()
                },
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Governance module not available",
            )

    @app.post("/governance/policy/check", response_model=PolicyCheckResponse)
    async def check_policy(
        request: PolicyCheckRequest,
        api_key: str = Depends(verify_api_key),
    ):
        """Check content and usage policies."""
        try:
            from axiom_vox.governance import get_policy_engine

            engine = get_policy_engine()
            results = engine.evaluate_all(
                text=request.text,
                operation=request.operation,
                voice_id=request.voice_id,
                is_commercial=request.is_commercial,
            )

            all_violations = []
            all_warnings = []
            policies_checked = []

            for r in results:
                all_violations.extend(r.violations)
                all_warnings.extend(r.warnings)
                policies_checked.append(r.policy_id)

            action = engine.get_action(
                text=request.text,
                operation=request.operation,
            )

            return PolicyCheckResponse(
                allowed=len(all_violations) == 0,
                action=action.value,
                violations=all_violations,
                warnings=all_warnings,
                policies_checked=policies_checked,
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Governance module not available",
            )

    @app.get("/governance/resources", response_model=ResourceStatusResponse)
    async def get_resource_status(
        api_key: str = Depends(verify_api_key),
    ):
        """Get current resource status."""
        try:
            from axiom_vox.governance import ResourceGovernor

            governor = ResourceGovernor()
            status = governor.get_status()

            return ResourceStatusResponse(
                resources=status["resources"],
                queue=status["queue"],
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Governance module not available",
            )

    @app.post("/governance/tier/{user_id}")
    async def set_user_tier(
        user_id: str,
        tier: str,
        api_key: str = Depends(verify_api_key),
    ):
        """Set tier for a user (admin operation)."""
        try:
            from axiom_vox.governance import (
                get_quota_manager,
                TieredRateLimiter,
                QUOTA_DEFAULTS,
            )

            if tier not in QUOTA_DEFAULTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid tier: {tier}. Valid tiers: {list(QUOTA_DEFAULTS.keys())}",
                )

            # Update quota manager
            quota_manager = get_quota_manager()
            quota_manager.set_user_tier(user_id, tier)

            # Update rate limiter
            rate_limiter = TieredRateLimiter()
            rate_limiter.set_user_tier(user_id, tier)

            return {
                "success": True,
                "user_id": user_id,
                "tier": tier,
                "message": f"User {user_id} set to tier {tier}",
            }

        except HTTPException:
            raise
        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Governance module not available",
            )

    @app.get("/governance/config", response_model=GovernanceConfigResponse)
    async def get_governance_config(
        api_key: str = Depends(verify_api_key),
    ):
        """Get current governance configuration."""
        try:
            from axiom_vox.governance import (
                RATE_LIMIT_DEFAULTS,
                QUOTA_DEFAULTS,
                RESOURCE_LIMITS,
            )

            return GovernanceConfigResponse(
                rate_limits={
                    tier: {
                        "requests_per_minute": cfg.requests_per_minute,
                        "requests_per_hour": cfg.requests_per_hour,
                        "requests_per_day": cfg.requests_per_day,
                        "burst_size": cfg.burst_size,
                    }
                    for tier, cfg in RATE_LIMIT_DEFAULTS.items()
                },
                quotas={
                    tier: {
                        "synthesis_per_day": cfg.synthesis_per_day,
                        "characters_per_day": cfg.characters_per_day,
                        "audio_seconds_per_day": cfg.audio_seconds_per_day,
                        "voice_slots": cfg.voice_slots,
                    }
                    for tier, cfg in QUOTA_DEFAULTS.items()
                },
                policies={
                    "content_filter": "enabled",
                    "usage_validator": "enabled",
                },
                security={
                    "api_key_required": require_api_key,
                    "rbac_enabled": True,
                    "audit_enabled": True,
                },
            )

        except ImportError:
            raise HTTPException(
                status_code=501,
                detail="Governance module not available",
            )

    return app


# ============================================================================
# STANDALONE SERVER
# ============================================================================

# Create default app for direct uvicorn usage
app = None
if HAS_FASTAPI:
    app = create_governed_tts_app(
        require_api_key=False,  # Disable for development
        strict_governance=False,
    )


if __name__ == "__main__":
    if not HAS_FASTAPI:
        print("FastAPI is required. Install with: pip install fastapi uvicorn")
        exit(1)

    import uvicorn

    print("=" * 70)
    print("  AXIØM VØX Governed TTS API")
    print("=" * 70)
    print("\nStarting server on http://0.0.0.0:8000")
    print("\nEndpoints:")
    print("  GET  /health             - Health check")
    print("  POST /check              - Pre-flight governance check")
    print("  POST /synthesize         - Governed TTS synthesis")
    print("  GET  /voices             - List available voices")
    print("  GET  /voice/{id}         - Voice details")
    print("  POST /clone              - Request voice clone")
    print("\nStreaming:")
    print("  POST /synthesize/stream  - HTTP streaming TTS")
    print("  WS   /ws/synthesize      - WebSocket streaming TTS")
    print("\nFine-tuning:")
    print("  POST   /finetune         - Start fine-tuning job")
    print("  GET    /finetune         - List fine-tuning jobs")
    print("  GET    /finetune/{id}    - Get job status")
    print("  DELETE /finetune/{id}    - Cancel job")
    print("\nDocs: http://localhost:8000/docs")
    print("-" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8000)
