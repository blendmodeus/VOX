"""
AXIOM VOX Analytics API Endpoints
---------------------------------

FastAPI router for voice analytics endpoints.

Endpoints:
- GET /analytics/summary - System-wide metrics
- GET /analytics/voice/{voice_id} - Per-voice metrics
- GET /analytics/trends - Time-series data
- GET /analytics/quality - Quality distribution
- GET /analytics/synthesis/{synthesis_id} - Single synthesis analytics
- GET /analytics/events - Recent analytics events
- POST /analytics/compute/{synthesis_id} - Trigger heavy computation

Usage:
    from axiom_vox.analytics import create_analytics_router

    router = create_analytics_router(analytics_service)
    app.include_router(router)
"""

import logging
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, Depends, HTTPException, Query
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None
    BaseModel = object

from axiom_vox.analytics.models import (
    AnalyticsEventType,
    QualityTier,
)
from axiom_vox.analytics.service import VoxAnalyticsService

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================

if FASTAPI_AVAILABLE:

    class AnalyticsSummaryResponse(BaseModel):
        """Response model for system analytics summary."""
        period: Dict[str, str]
        volume: Dict[str, Any]
        quality: Dict[str, float]
        performance: Dict[str, float]
        errors: Dict[str, Any]
        top_voices: List[Dict[str, Any]] = []

    class VoiceMetricsResponse(BaseModel):
        """Response model for voice metrics."""
        voice_id: str
        period: Dict[str, str]
        volume: Dict[str, Any]
        quality: Dict[str, float]
        performance: Dict[str, float]
        quality_distribution: Dict[str, int] = {}
        issues: Dict[str, int] = {}

    class TrendDataPoint(BaseModel):
        """Single trend data point."""
        date: str
        value: Optional[float]
        count: int

    class TrendAnalysis(BaseModel):
        """Trend analysis results."""
        direction: str
        change_percent: float
        anomalies: List[Dict[str, Any]] = []

    class TrendsResponse(BaseModel):
        """Response model for trend data."""
        metric: str
        voice_id: Optional[str]
        period_days: int
        granularity: str
        data: List[TrendDataPoint]
        trend: TrendAnalysis

    class QualityDistributionResponse(BaseModel):
        """Response model for quality distribution."""
        voice_id: Optional[str]
        period_days: int
        distribution: Dict[str, int]
        stats: Dict[str, float]

    class SynthesisAnalyticsResponse(BaseModel):
        """Response model for single synthesis analytics."""
        synthesis_id: str
        voice_id: str
        timestamp: float
        text_length: int
        quality_score: float
        quality_tier: str
        computation_status: str
        technical_quality: Optional[Dict[str, Any]] = None
        spectral_quality: Optional[Dict[str, Any]] = None
        naturalness: Optional[Dict[str, Any]] = None
        performance: Optional[Dict[str, Any]] = None
        governance_action: Optional[str] = None

    class AnalyticsEventResponse(BaseModel):
        """Response model for analytics event."""
        event_type: str
        timestamp: float
        voice_id: Optional[str]
        synthesis_id: Optional[str]
        severity: str
        message: str
        data: Dict[str, Any] = {}
        resolved: bool

    class ComputeResponse(BaseModel):
        """Response model for compute trigger."""
        success: bool
        synthesis_id: str
        message: str


# ============================================================================
# ROUTER FACTORY
# ============================================================================

def create_analytics_router(
    analytics_service: VoxAnalyticsService,
    prefix: str = "/analytics",
    tags: Optional[List[str]] = None,
    api_key_dependency: Optional[Any] = None,
) -> "APIRouter":
    """
    Create FastAPI router for analytics endpoints.

    Args:
        analytics_service: VoxAnalyticsService instance
        prefix: URL prefix for routes
        tags: OpenAPI tags
        api_key_dependency: Optional FastAPI dependency for auth

    Returns:
        APIRouter instance
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for analytics endpoints")

    router = APIRouter(prefix=prefix, tags=tags or ["analytics"])

    # Optional API key dependency
    dependencies = [Depends(api_key_dependency)] if api_key_dependency else []

    # ========================================================================
    # SUMMARY ENDPOINTS
    # ========================================================================

    @router.get(
        "/summary",
        response_model=AnalyticsSummaryResponse,
        dependencies=dependencies,
        summary="Get system-wide analytics summary",
        description="Returns aggregated metrics across all voices for the specified period.",
    )
    async def get_system_summary(
        period_days: int = Query(7, ge=1, le=365, description="Number of days to aggregate"),
    ) -> AnalyticsSummaryResponse:
        """Get system-wide analytics summary."""
        summary = await analytics_service.get_system_summary(period_days)
        return AnalyticsSummaryResponse(**summary.to_dict())

    @router.get(
        "/voice/{voice_id}",
        response_model=VoiceMetricsResponse,
        dependencies=dependencies,
        summary="Get voice-specific metrics",
        description="Returns aggregated metrics for a specific voice.",
    )
    async def get_voice_metrics(
        voice_id: str,
        period_days: int = Query(7, ge=1, le=365, description="Number of days to aggregate"),
    ) -> VoiceMetricsResponse:
        """Get metrics for a specific voice."""
        metrics = await analytics_service.get_voice_summary(voice_id, period_days)
        return VoiceMetricsResponse(**metrics.to_dict())

    # ========================================================================
    # TRENDS
    # ========================================================================

    @router.get(
        "/trends",
        response_model=TrendsResponse,
        dependencies=dependencies,
        summary="Get time-series trend data",
        description="Returns trend data for a specific metric over time.",
    )
    async def get_trends(
        metric: str = Query(..., description="Metric to trend (quality, latency, naturalness, mos, snr)"),
        voice_id: Optional[str] = Query(None, description="Optional voice filter"),
        period_days: int = Query(30, ge=1, le=365, description="Number of days"),
        granularity: str = Query("day", description="Time granularity (hour, day, week)"),
    ) -> TrendsResponse:
        """Get time-series trend data for a metric."""
        valid_metrics = {"quality", "latency", "naturalness", "mos", "snr"}
        if metric not in valid_metrics:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid metric. Must be one of: {valid_metrics}",
            )

        valid_granularities = {"hour", "day", "week"}
        if granularity not in valid_granularities:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid granularity. Must be one of: {valid_granularities}",
            )

        result = await analytics_service.get_trends(
            metric=metric,
            voice_id=voice_id,
            period_days=period_days,
            granularity=granularity,
        )

        return TrendsResponse(
            metric=result["metric"],
            voice_id=result["voice_id"],
            period_days=result["period_days"],
            granularity=result["granularity"],
            data=[TrendDataPoint(**d) for d in result["data"]],
            trend=TrendAnalysis(**result["trend"]),
        )

    # ========================================================================
    # QUALITY DISTRIBUTION
    # ========================================================================

    @router.get(
        "/quality",
        response_model=QualityDistributionResponse,
        dependencies=dependencies,
        summary="Get quality score distribution",
        description="Returns distribution of quality scores across tiers.",
    )
    async def get_quality_distribution(
        voice_id: Optional[str] = Query(None, description="Optional voice filter"),
        period_days: int = Query(7, ge=1, le=365, description="Number of days"),
    ) -> QualityDistributionResponse:
        """Get quality score distribution."""
        result = await analytics_service.get_quality_distribution(
            voice_id=voice_id,
            period_days=period_days,
        )
        return QualityDistributionResponse(**result)

    # ========================================================================
    # SINGLE SYNTHESIS
    # ========================================================================

    @router.get(
        "/synthesis/{synthesis_id}",
        response_model=SynthesisAnalyticsResponse,
        dependencies=dependencies,
        summary="Get analytics for a specific synthesis",
        description="Returns detailed analytics for a single synthesis operation.",
    )
    async def get_synthesis_analytics(
        synthesis_id: str,
    ) -> SynthesisAnalyticsResponse:
        """Get analytics for a specific synthesis."""
        analytics = analytics_service.storage.get_synthesis_metrics(synthesis_id)

        if analytics is None:
            raise HTTPException(
                status_code=404,
                detail=f"Synthesis not found: {synthesis_id}",
            )

        return SynthesisAnalyticsResponse(
            synthesis_id=analytics.synthesis_id,
            voice_id=analytics.voice_id,
            timestamp=analytics.timestamp,
            text_length=analytics.text_length,
            quality_score=analytics.get_quality_score(),
            quality_tier=analytics.get_quality_tier().value,
            computation_status=analytics.computation_status.value,
            technical_quality=analytics.technical_quality.to_dict() if analytics.technical_quality else None,
            spectral_quality=analytics.spectral_quality.to_dict() if analytics.spectral_quality else None,
            naturalness=analytics.naturalness.to_dict() if analytics.naturalness else None,
            performance=analytics.performance.to_dict() if analytics.performance else None,
            governance_action=analytics.governance_action,
        )

    # ========================================================================
    # EVENTS
    # ========================================================================

    @router.get(
        "/events",
        response_model=List[AnalyticsEventResponse],
        dependencies=dependencies,
        summary="Get recent analytics events",
        description="Returns recent notable events (quality drops, latency spikes, etc.).",
    )
    async def get_events(
        event_type: Optional[str] = Query(None, description="Filter by event type"),
        voice_id: Optional[str] = Query(None, description="Filter by voice ID"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
    ) -> List[AnalyticsEventResponse]:
        """Get recent analytics events."""
        # Parse event type
        event_type_enum = None
        if event_type:
            try:
                event_type_enum = AnalyticsEventType(event_type)
            except ValueError:
                valid_types = [e.value for e in AnalyticsEventType]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid event_type. Must be one of: {valid_types}",
                )

        events = await analytics_service.get_recent_events(
            event_type=event_type_enum,
            voice_id=voice_id,
            limit=limit,
        )

        return [
            AnalyticsEventResponse(
                event_type=e.event_type.value,
                timestamp=e.timestamp,
                voice_id=e.voice_id,
                synthesis_id=e.synthesis_id,
                severity=e.severity.value,
                message=e.message,
                data=e.data,
                resolved=e.resolved,
            )
            for e in events
        ]

    # ========================================================================
    # COMPUTE TRIGGER
    # ========================================================================

    @router.post(
        "/compute/{synthesis_id}",
        response_model=ComputeResponse,
        dependencies=dependencies,
        summary="Trigger heavy metric computation",
        description="Queues heavy computation (naturalness) for a synthesis.",
    )
    async def trigger_computation(
        synthesis_id: str,
    ) -> ComputeResponse:
        """Trigger heavy metric computation for a synthesis."""
        # Check if synthesis exists
        analytics = analytics_service.storage.get_synthesis_metrics(synthesis_id)
        if analytics is None:
            raise HTTPException(
                status_code=404,
                detail=f"Synthesis not found: {synthesis_id}",
            )

        # Note: In practice, we'd need stored audio data to compute naturalness
        # This endpoint would typically be used with a separate audio storage system

        return ComputeResponse(
            success=True,
            synthesis_id=synthesis_id,
            message="Computation queued (requires stored audio data)",
        )

    # ========================================================================
    # STREAMING ANALYTICS ENDPOINTS
    # ========================================================================

    @router.get(
        "/streaming/session/{session_id}",
        dependencies=dependencies,
        summary="Get streaming session analytics",
        description="Returns detailed streaming metrics for a session.",
    )
    async def get_streaming_session(
        session_id: str,
    ) -> Dict[str, Any]:
        """Get analytics for a streaming session."""
        session = analytics_service.storage.get_streaming_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Streaming session not found: {session_id}",
            )
        return session

    @router.get(
        "/streaming/summary",
        dependencies=dependencies,
        summary="Get streaming performance summary",
        description="Returns aggregated streaming metrics.",
    )
    async def get_streaming_summary(
        voice_id: Optional[str] = Query(None, description="Filter by voice"),
        period_days: int = Query(7, ge=1, le=365, description="Period in days"),
    ) -> Dict[str, Any]:
        """Get streaming performance summary."""
        return analytics_service.storage.get_streaming_summary(
            voice_id=voice_id,
            period_days=period_days,
        )

    @router.get(
        "/streaming/live/{session_id}",
        dependencies=dependencies,
        summary="Get live streaming metrics",
        description="Returns current metrics for an active streaming session.",
    )
    async def get_live_streaming_metrics(
        session_id: str,
    ) -> Dict[str, Any]:
        """Get live metrics for an active streaming session."""
        try:
            from axiom_vox.analytics.streaming_collector import get_streaming_collector

            collector = get_streaming_collector()
            metrics = collector.get_current_metrics(session_id)

            if not metrics:
                raise HTTPException(
                    status_code=404,
                    detail=f"Active session not found: {session_id}",
                )

            return {
                "session_id": session_id,
                "is_active": True,
                "metrics": metrics.to_dict(),
                "recent_chunks": [
                    {
                        "index": c.chunk_index,
                        "latency_ms": c.latency_ms,
                        "is_late": c.is_late,
                    }
                    for c in collector.get_recent_chunks(session_id, count=10)
                ],
            }
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Streaming analytics not available",
            )

    @router.get(
        "/streaming/events",
        dependencies=dependencies,
        summary="Get streaming events",
        description="Returns recent streaming events (stutters, late chunks).",
    )
    async def get_streaming_events(
        session_id: Optional[str] = Query(None, description="Filter by session"),
        event_type: Optional[str] = Query(None, description="Filter by event type"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
    ) -> List[Dict[str, Any]]:
        """Get streaming events."""
        return analytics_service.storage.get_streaming_events(
            session_id=session_id,
            event_type=event_type,
            limit=limit,
        )

    # ========================================================================
    # HEALTH CHECK
    # ========================================================================

    @router.get(
        "/health",
        dependencies=dependencies,
        summary="Analytics health check",
        description="Returns analytics service health status.",
    )
    async def health_check() -> Dict[str, Any]:
        """Check analytics service health."""
        active_streams = 0
        try:
            from axiom_vox.analytics.streaming_collector import get_streaming_collector
            active_streams = get_streaming_collector().get_active_session_count()
        except ImportError:
            pass

        return {
            "status": "healthy",
            "workers_running": len(analytics_service._workers) > 0,
            "queue_size": analytics_service._compute_queue.qsize(),
            "active_streams": active_streams,
        }

    return router


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  AXIOM VOX Analytics Endpoints")
    print("=" * 70)

    if FASTAPI_AVAILABLE:
        print("\nFastAPI available - endpoints can be created")
        print("\nAvailable endpoints:")
        print("  GET  /analytics/summary")
        print("  GET  /analytics/voice/{voice_id}")
        print("  GET  /analytics/trends")
        print("  GET  /analytics/quality")
        print("  GET  /analytics/synthesis/{synthesis_id}")
        print("  GET  /analytics/events")
        print("  POST /analytics/compute/{synthesis_id}")
        print("  GET  /analytics/streaming/session/{session_id}")
        print("  GET  /analytics/streaming/summary")
        print("  GET  /analytics/streaming/live/{session_id}")
        print("  GET  /analytics/streaming/events")
        print("  GET  /analytics/health")
    else:
        print("\nFastAPI not available - install with: pip install fastapi")

    print("\n" + "=" * 70)
