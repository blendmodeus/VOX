"""
AXIØM VØX Unified Pipeline
--------------------------

Single entry point for all voice operations.

AXIØM Phase 6: System - "Integrate the parts"

v0.11.0: Unified Voice Pipeline

Components:
    - VoxUnifiedPipeline: Orchestrates all voice operations
    - UnifiedConsentRegistry: Central consent management
    - BiometricVoiceRouter: Intelligent voice selection
    - RealTimeQualityMonitor: Live quality assessment

Usage:
    from axiom_vox.unified import VoxUnifiedPipeline, PipelineRequest

    # Create pipeline
    pipeline = VoxUnifiedPipeline()

    # Process request
    request = PipelineRequest(
        text="Hello, world!",
        voice_id="my_voice",
    )
    response = await pipeline.process(request)

    # Stream synthesis
    async for chunk in pipeline.process_stream(request):
        print(chunk)
"""

# Models
from .models import (
    # Enums
    PipelineStage,
    PipelineStatus,
    VoiceRouteType,
    ConsentScope,
    QualityGate,
    # Dataclasses
    PipelineRequest,
    PipelineResponse,
    PipelineConfig,
    PipelineMetrics,
    StageResult,
    IdentityResult,
    ConsentResult,
    GovernanceResult,
    RouteResult,
    SynthesisResult,
    QualityResult,
    VoiceProfile,
    # Constants
    STAGE_ORDER,
    QUALITY_THRESHOLDS,
)

# Consent Registry
from .consent_registry import (
    UnifiedConsentRegistry,
    ConsentRecord,
    ConsentQuery,
    get_consent_registry,
    set_consent_registry,
)

# Voice Router
from .voice_router import (
    BiometricVoiceRouter,
    RouteCandidate,
    get_voice_router,
    set_voice_router,
)

# Quality Monitor
from .quality_monitor import (
    RealTimeQualityMonitor,
    QualityEvent,
    QualitySnapshot,
    QualityAlert,
    get_quality_monitor,
    set_quality_monitor,
)

# Pipeline
from .pipeline import (
    VoxUnifiedPipeline,
    get_unified_pipeline,
    set_unified_pipeline,
    synthesize_unified,
)

__all__ = [
    # Models - Enums
    "PipelineStage",
    "PipelineStatus",
    "VoiceRouteType",
    "ConsentScope",
    "QualityGate",
    # Models - Dataclasses
    "PipelineRequest",
    "PipelineResponse",
    "PipelineConfig",
    "PipelineMetrics",
    "StageResult",
    "IdentityResult",
    "ConsentResult",
    "GovernanceResult",
    "RouteResult",
    "SynthesisResult",
    "QualityResult",
    "VoiceProfile",
    # Models - Constants
    "STAGE_ORDER",
    "QUALITY_THRESHOLDS",
    # Consent Registry
    "UnifiedConsentRegistry",
    "ConsentRecord",
    "ConsentQuery",
    "get_consent_registry",
    "set_consent_registry",
    # Voice Router
    "BiometricVoiceRouter",
    "RouteCandidate",
    "get_voice_router",
    "set_voice_router",
    # Quality Monitor
    "RealTimeQualityMonitor",
    "QualityEvent",
    "QualitySnapshot",
    "QualityAlert",
    "get_quality_monitor",
    "set_quality_monitor",
    # Pipeline
    "VoxUnifiedPipeline",
    "get_unified_pipeline",
    "set_unified_pipeline",
    "synthesize_unified",
]

__version__ = "0.11.0"
