"""
AXIOM VOX Analytics
-------------------

Comprehensive voice analytics for synthesis quality monitoring.

Components:
- Models: Dataclasses for metrics (technical, spectral, naturalness, performance)
- Analyzer: Audio quality analysis utilities
- Collector: Synchronous metrics collection during synthesis
- Storage: SQLite persistence layer
- Service: Aggregation, trends, and async computation
- Endpoints: FastAPI router for analytics API

Quick Start:
    from axiom_vox.analytics import (
        VoxMetricsCollector,
        VoxAnalyticsService,
        AnalyticsStorage,
        SynthesisAnalytics,
    )

    # Create collector for synthesis
    collector = VoxMetricsCollector()
    collector.start_synthesis("synth_123", "Hello world", "voice_1")
    # ... synthesis ...
    perf = collector.end_synthesis("synth_123")
    tech, spectral = collector.analyze_audio("synth_123", audio_bytes, 24000)
    analytics = collector.get_analytics("synth_123")

    # Store and aggregate
    storage = AnalyticsStorage()
    storage.save_synthesis_metrics(analytics)
    summary = storage.get_system_summary(period_days=7)

Usage with API:
    from axiom_vox.analytics import create_analytics_router, init_analytics_service

    service = init_analytics_service()
    router = create_analytics_router(service)
    app.include_router(router)
"""

# Models
from axiom_vox.analytics.models import (
    # Enums
    ComputationStatus,
    QualityTier,
    AnalyticsEventType,
    EventSeverity,
    # Quality metrics
    TechnicalQualityMetrics,
    SpectralQualityMetrics,
    NaturalnessMetrics,
    PerformanceMetrics,
    VoiceConsistencyMetrics,
    # Analytics containers
    SynthesisAnalytics,
    StreamingSessionAnalytics,
    VoiceAggregateMetrics,
    SystemAnalyticsSummary,
    AnalyticsEvent,
)

# Analyzer
from axiom_vox.analytics.analyzer import VoxAudioAnalyzer

# Collector
from axiom_vox.analytics.collector import (
    VoxMetricsCollector,
    get_collector,
    set_collector,
)

# Storage
from axiom_vox.analytics.storage import AnalyticsStorage

# Service
from axiom_vox.analytics.service import (
    VoxAnalyticsService,
    get_analytics_service,
    init_analytics_service,
)

# Streaming Analytics
from axiom_vox.analytics.streaming_collector import (
    StreamingAnalyticsCollector,
    StreamChunkMetrics,
    StreamSessionMetrics,
    StreamingEvent,
    StreamingEventType,
    get_streaming_collector,
    set_streaming_collector,
)

# Endpoints (conditional import)
try:
    from axiom_vox.analytics.endpoints import create_analytics_router
except ImportError:
    # FastAPI not available
    def create_analytics_router(*args, **kwargs):
        raise ImportError("FastAPI required for analytics endpoints: pip install fastapi")


__all__ = [
    # Enums
    "ComputationStatus",
    "QualityTier",
    "AnalyticsEventType",
    "EventSeverity",
    # Quality metrics
    "TechnicalQualityMetrics",
    "SpectralQualityMetrics",
    "NaturalnessMetrics",
    "PerformanceMetrics",
    "VoiceConsistencyMetrics",
    # Analytics containers
    "SynthesisAnalytics",
    "StreamingSessionAnalytics",
    "VoiceAggregateMetrics",
    "SystemAnalyticsSummary",
    "AnalyticsEvent",
    # Analyzer
    "VoxAudioAnalyzer",
    # Collector
    "VoxMetricsCollector",
    "get_collector",
    "set_collector",
    # Streaming Analytics
    "StreamingAnalyticsCollector",
    "StreamChunkMetrics",
    "StreamSessionMetrics",
    "StreamingEvent",
    "StreamingEventType",
    "get_streaming_collector",
    "set_streaming_collector",
    # Storage
    "AnalyticsStorage",
    # Service
    "VoxAnalyticsService",
    "get_analytics_service",
    "init_analytics_service",
    # Endpoints
    "create_analytics_router",
]

__version__ = "0.2.0"  # Added streaming analytics
