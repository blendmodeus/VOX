"""
VØX Unified Pipeline
--------------------

Single entry point for all voice operations.

Orchestrates:
    1. Intake - Request validation
    2. Identity - Biometric identification/verification
    3. Consent - Unified consent check
    4. Governance - AXIOM governance
    5. Routing - Voice selection
    6. Synthesis - Audio generation
    7. Quality - Quality assessment
    8. Delivery - Output formatting

AXIØM Phase 6: System - "Integrate the parts"
"""

import asyncio
import base64
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncIterator

import numpy as np

from .models import (
    PipelineStage,
    PipelineStatus,
    PipelineRequest,
    PipelineResponse,
    PipelineConfig,
    PipelineMetrics,
    StageResult,
    IdentityResult,
    GovernanceResult,
    SynthesisResult,
    ConsentScope,
    STAGE_ORDER,
)
from .consent_registry import UnifiedConsentRegistry, get_consent_registry
from .voice_router import BiometricVoiceRouter, get_voice_router
from .quality_monitor import RealTimeQualityMonitor, get_quality_monitor

logger = logging.getLogger(__name__)


class VoxUnifiedPipeline:
    """
    Unified voice pipeline orchestrating all VØX components.

    Provides single entry point for:
        - Text-to-speech synthesis
        - Voice identification/verification
        - Multi-voice dialogue
        - Streaming synthesis
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize unified pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Component instances (lazy loaded)
        self._consent_registry: Optional[UnifiedConsentRegistry] = None
        self._voice_router: Optional[BiometricVoiceRouter] = None
        self._quality_monitor: Optional[RealTimeQualityMonitor] = None
        self._governor = None
        self._synthesizer = None

    @property
    def consent_registry(self) -> UnifiedConsentRegistry:
        """Get consent registry."""
        if self._consent_registry is None:
            self._consent_registry = get_consent_registry()
        return self._consent_registry

    @property
    def voice_router(self) -> BiometricVoiceRouter:
        """Get voice router."""
        if self._voice_router is None:
            self._voice_router = get_voice_router(self.config)
        return self._voice_router

    @property
    def quality_monitor(self) -> RealTimeQualityMonitor:
        """Get quality monitor."""
        if self._quality_monitor is None:
            self._quality_monitor = get_quality_monitor(self.config)
        return self._quality_monitor

    async def process(
        self,
        request: PipelineRequest,
    ) -> PipelineResponse:
        """
        Process a voice synthesis request through the unified pipeline.

        Args:
            request: Pipeline request

        Returns:
            PipelineResponse with complete results
        """
        response = PipelineResponse(
            request_id=request.request_id,
            status=PipelineStatus.PROCESSING,
            started_at=time.time(),
        )

        metrics = PipelineMetrics(request_id=request.request_id)

        try:
            # Stage 1: Intake
            intake_result = await self._stage_intake(request)
            response.stage_results[PipelineStage.INTAKE.value] = intake_result
            response.stages_completed.append(PipelineStage.INTAKE)

            if not intake_result.success:
                response.status = PipelineStatus.FAILED
                response.error = intake_result.error
                return self._finalize_response(response, metrics)

            # Stage 2: Identity (if audio input or biometric required)
            if request.audio_input or request.require_biometric_verification:
                identity_result = await self._stage_identity(request)
                response.stage_results[PipelineStage.IDENTITY.value] = identity_result
                response.stages_completed.append(PipelineStage.IDENTITY)
                response.identity = identity_result.data.get("identity")
                metrics.biometric_checked = True
                metrics.biometric_verified = identity_result.success

            # Stage 3: Consent
            if self.config.require_consent:
                consent_result = await self._stage_consent(request, response.identity)
                response.stage_results[PipelineStage.CONSENT.value] = consent_result
                response.stages_completed.append(PipelineStage.CONSENT)
                response.consent = consent_result.data.get("consent")
                metrics.consent_checked = True
                metrics.consent_granted = consent_result.success

                if not consent_result.success:
                    response.status = PipelineStatus.BLOCKED
                    response.error = consent_result.error
                    return self._finalize_response(response, metrics)

            # Stage 4: Governance
            governance_result = await self._stage_governance(request)
            response.stage_results[PipelineStage.GOVERNANCE.value] = governance_result
            response.stages_completed.append(PipelineStage.GOVERNANCE)
            response.governance = governance_result.data.get("governance")

            if not governance_result.success and self.config.strict_governance:
                response.status = PipelineStatus.BLOCKED
                response.error = governance_result.error
                return self._finalize_response(response, metrics)

            # Get governed text
            governed_text = (
                response.governance.governed_text
                if response.governance
                else request.text or ""
            )
            response.governed_text = governed_text

            # Stage 5: Routing
            routing_result = await self._stage_routing(
                request,
                response.identity,
            )
            response.stage_results[PipelineStage.ROUTING.value] = routing_result
            response.stages_completed.append(PipelineStage.ROUTING)
            response.route = routing_result.data.get("route")
            response.voice_id_used = response.route.voice_id if response.route else None
            metrics.voice_id = response.voice_id_used or ""
            metrics.route_type = response.route.route_type.value if response.route else ""

            # Stage 6: Synthesis
            synthesis_result = await self._stage_synthesis(
                governed_text,
                response.route,
                request,
            )
            response.stage_results[PipelineStage.SYNTHESIS.value] = synthesis_result
            response.stages_completed.append(PipelineStage.SYNTHESIS)
            response.synthesis = synthesis_result.data.get("synthesis")
            metrics.synthesis_latency_ms = synthesis_result.duration_ms

            if not synthesis_result.success:
                response.status = PipelineStatus.FAILED
                response.error = synthesis_result.error
                return self._finalize_response(response, metrics)

            # Stage 7: Quality
            if self.config.enable_quality_gates and response.synthesis:
                quality_result = await self._stage_quality(response.synthesis)
                response.stage_results[PipelineStage.QUALITY.value] = quality_result
                response.stages_completed.append(PipelineStage.QUALITY)
                response.quality = quality_result.data.get("quality")
                metrics.quality_score = response.quality.overall_score if response.quality else 0

                if self.config.block_on_quality_failure and not quality_result.success:
                    response.status = PipelineStatus.FAILED
                    response.error = "Quality gate failed"
                    return self._finalize_response(response, metrics)

            # Stage 8: Delivery
            delivery_result = await self._stage_delivery(response)
            response.stage_results[PipelineStage.DELIVERY.value] = delivery_result
            response.stages_completed.append(PipelineStage.DELIVERY)

            # Set final output
            if response.synthesis and response.synthesis.audio_data:
                response.audio_data = response.synthesis.audio_data
                response.audio_base64 = base64.b64encode(
                    response.synthesis.audio_data
                ).decode()

            response.status = PipelineStatus.COMPLETED
            metrics.success = True
            metrics.audio_duration_seconds = (
                response.synthesis.duration_seconds if response.synthesis else 0
            )

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            response.status = PipelineStatus.FAILED
            response.error = str(e)
            metrics.error_message = str(e)

        return self._finalize_response(response, metrics)

    async def process_stream(
        self,
        request: PipelineRequest,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process request with streaming output.

        Yields:
            Stream messages with audio chunks and metadata
        """
        request.stream = True
        request_id = request.request_id

        # Run pre-synthesis stages
        yield {"type": "start", "request_id": request_id}

        try:
            # Intake
            intake_result = await self._stage_intake(request)
            if not intake_result.success:
                yield {"type": "error", "error": intake_result.error}
                return

            # Identity (if needed)
            identity = None
            if request.audio_input or request.require_biometric_verification:
                identity_result = await self._stage_identity(request)
                identity = identity_result.data.get("identity")
                yield {
                    "type": "identity",
                    "verified": identity.verified if identity else False,
                }

            # Consent
            if self.config.require_consent:
                consent_result = await self._stage_consent(request, identity)
                if not consent_result.success:
                    yield {"type": "blocked", "reason": "consent", "message": consent_result.error}
                    return

            # Governance
            governance_result = await self._stage_governance(request)
            governance = governance_result.data.get("governance")
            governed_text = governance.governed_text if governance else request.text

            yield {
                "type": "governance",
                "approved": governance_result.success,
                "governed_text": governed_text,
            }

            if not governance_result.success and self.config.strict_governance:
                yield {"type": "blocked", "reason": "governance"}
                return

            # Routing
            routing_result = await self._stage_routing(request, identity)
            route = routing_result.data.get("route")

            yield {
                "type": "routing",
                "voice_id": route.voice_id if route else self.config.default_voice_id,
                "route_type": route.route_type.value if route else "default",
            }

            # Stream synthesis
            self.quality_monitor.start_session(request_id)

            async for chunk in self._stream_synthesis(governed_text, route, request):
                # Analyze quality
                if chunk.get("audio") and self.config.enable_streaming_quality:
                    audio_array = np.frombuffer(chunk["audio"], dtype=np.float32)
                    snapshot = self.quality_monitor.analyze_chunk(audio_array)
                    chunk["quality_score"] = snapshot.overall_score

                yield chunk

            # Finalize quality
            quality = self.quality_monitor.finalize_session()

            yield {
                "type": "complete",
                "quality": {
                    "overall_score": quality.overall_score,
                    "gate_status": quality.gate_status.value,
                },
            }

        except Exception as e:
            logger.exception(f"Stream error: {e}")
            yield {"type": "error", "error": str(e)}

    async def _stage_intake(self, request: PipelineRequest) -> StageResult:
        """Validate and process incoming request."""
        start = time.time()

        try:
            # Validate input
            if not request.text and not request.ssml:
                return StageResult(
                    stage=PipelineStage.INTAKE,
                    success=False,
                    error="No text or SSML provided",
                    duration_ms=(time.time() - start) * 1000,
                )

            # Check text length
            text = request.text or ""
            if len(text) > self.config.max_text_length:
                return StageResult(
                    stage=PipelineStage.INTAKE,
                    success=False,
                    error=f"Text exceeds max length ({len(text)} > {self.config.max_text_length})",
                    duration_ms=(time.time() - start) * 1000,
                )

            return StageResult(
                stage=PipelineStage.INTAKE,
                success=True,
                message="Request validated",
                duration_ms=(time.time() - start) * 1000,
                data={"text_length": len(text)},
            )

        except Exception as e:
            return StageResult(
                stage=PipelineStage.INTAKE,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def _stage_identity(self, request: PipelineRequest) -> StageResult:
        """Identify/verify speaker from audio."""
        start = time.time()

        try:
            identity = IdentityResult()

            if request.speaker_embedding:
                # Use provided embedding
                identity.speaker_embedding = request.speaker_embedding
                identity.identified = True

            elif request.audio_input:
                # Extract embedding from audio
                try:
                    from ..biometrics import SpectralFingerprint, serialize_embedding

                    audio = np.frombuffer(request.audio_input, dtype=np.float32)
                    fp = SpectralFingerprint()
                    embedding = fp.extract(audio, 24000)
                    identity.speaker_embedding = serialize_embedding(embedding)
                    identity.identified = True

                    # Try to match to enrolled voice
                    from ..biometrics import get_biometric_storage, deserialize_embedding

                    storage = get_biometric_storage()
                    enrolled = storage.list_enrolled_voices(limit=50)

                    for entry in enrolled:
                        template = storage.get_template(entry["voice_id"])
                        if template:
                            template_emb = deserialize_embedding(template.embedding)
                            similarity = fp.similarity(embedding, template_emb)
                            if similarity >= self.config.biometric_similarity_threshold:
                                identity.voice_id = entry["voice_id"]
                                identity.verified = True
                                identity.similarity_score = similarity
                                break

                except ImportError:
                    logger.debug("Biometrics not available for identity")

            return StageResult(
                stage=PipelineStage.IDENTITY,
                success=identity.identified,
                message=identity.message or "Identity processed",
                duration_ms=(time.time() - start) * 1000,
                data={"identity": identity},
            )

        except Exception as e:
            return StageResult(
                stage=PipelineStage.IDENTITY,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def _stage_consent(
        self,
        request: PipelineRequest,
        identity: Optional[IdentityResult],
    ) -> StageResult:
        """Check consent for requested operation."""
        start = time.time()

        try:
            voice_id = request.voice_id or (identity.voice_id if identity else None)
            if not voice_id:
                voice_id = self.config.default_voice_id

            # Determine required scopes
            required_scopes = [ConsentScope.SYNTHESIS]
            if request.stream:
                required_scopes.append(ConsentScope.STREAMING)

            result = self.consent_registry.check_consent(
                voice_id=voice_id,
                required_scopes=required_scopes,
                user_id=request.user_id,
            )

            return StageResult(
                stage=PipelineStage.CONSENT,
                success=result.granted,
                message=result.message,
                duration_ms=(time.time() - start) * 1000,
                data={"consent": result},
                warnings=result.restrictions,
            )

        except Exception as e:
            # On error, allow if not strict
            return StageResult(
                stage=PipelineStage.CONSENT,
                success=not self.config.require_consent,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def _stage_governance(self, request: PipelineRequest) -> StageResult:
        """Apply AXIOM governance."""
        start = time.time()

        try:
            from ..vox_governor import VoxGovernor, GovernanceAction

            if self._governor is None:
                self._governor = VoxGovernor(strict_mode=self.config.strict_governance)

            text = request.text or ""

            result = self._governor.govern(
                text=text,
                voice_id=request.voice_id,
                context=request.context,
            )

            governance = GovernanceResult(
                approved=result.action != GovernanceAction.REFUSE,
                action=result.action.value,
                original_text=text,
                governed_text=result.governed_text or text,
                repairs_made=result.repairs_made,
                violations=[v.message for v in (result.violations or [])],
                warnings=result.warnings,
            )

            return StageResult(
                stage=PipelineStage.GOVERNANCE,
                success=governance.approved,
                message=result.refusal_reason or "Governance passed",
                duration_ms=(time.time() - start) * 1000,
                data={"governance": governance},
                warnings=governance.warnings,
            )

        except Exception as e:
            # On error, pass through with original text
            governance = GovernanceResult(
                approved=True,
                original_text=request.text or "",
                governed_text=request.text or "",
            )
            return StageResult(
                stage=PipelineStage.GOVERNANCE,
                success=True,
                message=f"Governance error (passed): {e}",
                duration_ms=(time.time() - start) * 1000,
                data={"governance": governance},
            )

    async def _stage_routing(
        self,
        request: PipelineRequest,
        identity: Optional[IdentityResult],
    ) -> StageResult:
        """Route to optimal voice."""
        start = time.time()

        try:
            route = await self.voice_router.route(
                voice_id=request.voice_id,
                speaker_embedding=identity.speaker_embedding if identity else None,
                text=request.text,
                context=request.context,
                require_verification=request.require_biometric_verification,
            )

            return StageResult(
                stage=PipelineStage.ROUTING,
                success=True,
                message=route.message,
                duration_ms=(time.time() - start) * 1000,
                data={"route": route},
            )

        except Exception as e:
            # Fallback to default voice
            from .models import RouteResult, VoiceRouteType

            route = RouteResult(
                voice_id=self.config.default_voice_id,
                route_type=VoiceRouteType.DEFAULT,
                fallback_used=True,
                message=f"Routing error, using default: {e}",
            )
            return StageResult(
                stage=PipelineStage.ROUTING,
                success=True,
                message=route.message,
                duration_ms=(time.time() - start) * 1000,
                data={"route": route},
            )

    async def _stage_synthesis(
        self,
        text: str,
        route,
        request: PipelineRequest,
    ) -> StageResult:
        """Synthesize audio."""
        start = time.time()

        try:
            from ..synthesis import VoxSynthesizer, VoiceConfig

            if self._synthesizer is None:
                self._synthesizer = VoxSynthesizer()

            voice_config = VoiceConfig(
                voice_id=route.voice_id if route else self.config.default_voice_id,
                speaking_rate=request.speaking_rate,
                emotion=request.emotion_preset,
            )

            # Load adapter if cloned voice
            if route and route.adapter_path:
                self._synthesizer.load_adapter(route.adapter_path)

            result = await asyncio.to_thread(
                self._synthesizer.synthesize,
                text=text,
                voice_config=voice_config,
            )

            synthesis = SynthesisResult(
                audio_data=result.audio_data,
                audio_format=result.format.value if hasattr(result, 'format') else "wav",
                duration_seconds=result.duration_seconds,
                sample_rate=result.sample_rate,
                synthesis_time_ms=(time.time() - start) * 1000,
            )

            if synthesis.duration_seconds > 0:
                synthesis.rtf = synthesis.synthesis_time_ms / 1000 / synthesis.duration_seconds

            return StageResult(
                stage=PipelineStage.SYNTHESIS,
                success=True,
                message="Synthesis complete",
                duration_ms=synthesis.synthesis_time_ms,
                data={"synthesis": synthesis},
            )

        except Exception as e:
            return StageResult(
                stage=PipelineStage.SYNTHESIS,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def _stage_quality(self, synthesis: SynthesisResult) -> StageResult:
        """Assess synthesis quality."""
        start = time.time()

        try:
            if not synthesis.audio_data:
                return StageResult(
                    stage=PipelineStage.QUALITY,
                    success=True,
                    message="No audio to assess",
                    duration_ms=(time.time() - start) * 1000,
                )

            # Start quality session
            session_id = f"quality_{uuid.uuid4().hex[:8]}"
            self.quality_monitor.start_session(session_id)

            # Analyze full audio
            audio = np.frombuffer(synthesis.audio_data, dtype=np.float32)
            chunk_size = synthesis.sample_rate  # 1 second chunks

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                self.quality_monitor.analyze_chunk(chunk, synthesis.sample_rate)

            quality = self.quality_monitor.finalize_session()

            success = quality.gate_status.value in ["passed", "warning"]

            return StageResult(
                stage=PipelineStage.QUALITY,
                success=success,
                message=f"Quality: {quality.overall_score:.3f}",
                duration_ms=(time.time() - start) * 1000,
                data={"quality": quality},
                warnings=quality.issues,
            )

        except Exception as e:
            return StageResult(
                stage=PipelineStage.QUALITY,
                success=True,  # Don't fail on quality errors
                message=f"Quality check error: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def _stage_delivery(self, response: PipelineResponse) -> StageResult:
        """Prepare final delivery."""
        start = time.time()

        # Nothing special for now, just finalize
        return StageResult(
            stage=PipelineStage.DELIVERY,
            success=True,
            message="Delivery prepared",
            duration_ms=(time.time() - start) * 1000,
        )

    async def _stream_synthesis(
        self,
        text: str,
        route,
        request: PipelineRequest,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream synthesis chunks."""
        try:
            from ..synthesis import VoxSynthesizer, VoiceConfig

            if self._synthesizer is None:
                self._synthesizer = VoxSynthesizer()

            voice_config = VoiceConfig(
                voice_id=route.voice_id if route else self.config.default_voice_id,
                speaking_rate=request.speaking_rate,
            )

            chunk_index = 0
            async for chunk in self._synthesizer.synthesize_stream(text, voice_config):
                yield {
                    "type": "chunk",
                    "index": chunk_index,
                    "audio": chunk.data if hasattr(chunk, 'data') else chunk,
                }
                chunk_index += 1

        except Exception as e:
            yield {"type": "error", "error": str(e)}

    def _finalize_response(
        self,
        response: PipelineResponse,
        metrics: PipelineMetrics,
    ) -> PipelineResponse:
        """Finalize response with timing and metrics."""
        response.completed_at = time.time()
        response.total_duration_ms = (response.completed_at - response.started_at) * 1000

        # Collect all warnings
        for stage_result in response.stage_results.values():
            response.warnings.extend(stage_result.warnings)

        # Log metrics
        metrics.total_duration_ms = response.total_duration_ms
        for stage, result in response.stage_results.items():
            metrics.stage_durations[stage] = result.duration_ms

        logger.info(
            f"Pipeline {response.request_id}: status={response.status.value}, "
            f"duration={response.total_duration_ms:.1f}ms"
        )

        return response


# Singleton instance
_pipeline_instance: Optional[VoxUnifiedPipeline] = None


def get_unified_pipeline(
    config: Optional[PipelineConfig] = None,
) -> VoxUnifiedPipeline:
    """Get or create unified pipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = VoxUnifiedPipeline(config=config)
    return _pipeline_instance


def set_unified_pipeline(pipeline: VoxUnifiedPipeline) -> None:
    """Set the unified pipeline singleton."""
    global _pipeline_instance
    _pipeline_instance = pipeline


# Convenience function
async def synthesize_unified(
    text: str,
    voice_id: Optional[str] = None,
    **kwargs,
) -> PipelineResponse:
    """
    Synthesize using unified pipeline.

    Args:
        text: Text to synthesize
        voice_id: Optional voice ID
        **kwargs: Additional request parameters

    Returns:
        PipelineResponse
    """
    pipeline = get_unified_pipeline()
    request = PipelineRequest(
        text=text,
        voice_id=voice_id,
        **kwargs,
    )
    return await pipeline.process(request)
