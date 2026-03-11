"""
AXIOM VOX Analytics Service
---------------------------

Analytics service for aggregation and async computation.

Manages:
- Background workers for heavy computation (naturalness)
- Metric aggregation by voice and time period
- Trend analysis
- Event detection and alerting

Usage:
    from axiom_vox.analytics import VoxAnalyticsService, AnalyticsStorage

    storage = AnalyticsStorage()
    service = VoxAnalyticsService(storage)
    service.start_workers()

    # Record synthesis
    await service.record_synthesis(analytics)

    # Get summaries
    summary = await service.get_system_summary(period_days=7)
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

from axiom_vox.analytics.models import (
    SynthesisAnalytics,
    VoiceAggregateMetrics,
    SystemAnalyticsSummary,
    AnalyticsEvent,
    AnalyticsEventType,
    EventSeverity,
    ComputationStatus,
)
from axiom_vox.analytics.storage import AnalyticsStorage
from axiom_vox.analytics.analyzer import VoxAudioAnalyzer

logger = logging.getLogger(__name__)


class VoxAnalyticsService:
    """
    Analytics service for aggregation and async computation.

    Provides:
    - Async recording of synthesis analytics
    - Background workers for heavy computation
    - Summary and trend endpoints
    - Event detection for anomalies
    """

    def __init__(
        self,
        storage: AnalyticsStorage,
        num_workers: int = 2,
        batch_size: int = 10,
        enable_event_detection: bool = True,
    ):
        """
        Initialize analytics service.

        Args:
            storage: AnalyticsStorage instance for persistence
            num_workers: Number of background worker threads
            batch_size: Batch size for aggregation operations
            enable_event_detection: Enable automatic event detection
        """
        self.storage = storage
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.enable_event_detection = enable_event_detection

        self._analyzer = VoxAudioAnalyzer()
        self._compute_queue: Queue = Queue()
        self._workers: List[threading.Thread] = []
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

        # Thresholds for event detection
        self._quality_threshold = 0.4  # Alert if quality drops below
        self._latency_threshold_ms = 1000  # Alert if latency exceeds
        self._artifact_threshold = 0.3  # Alert if artifact score exceeds

    # ========================================================================
    # WORKER MANAGEMENT
    # ========================================================================

    def start_workers(self) -> None:
        """Start background worker threads."""
        if self._running:
            logger.warning("Workers already running")
            return

        self._running = True

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"analytics-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        logger.info(f"Started {self.num_workers} analytics workers")

    def stop_workers(self, timeout: float = 5.0) -> None:
        """
        Stop background workers gracefully.

        Args:
            timeout: Maximum time to wait for workers to finish
        """
        self._running = False

        # Signal workers to stop
        for _ in self._workers:
            self._compute_queue.put(None)

        # Wait for workers
        for worker in self._workers:
            worker.join(timeout=timeout)

        self._workers.clear()
        logger.info("Analytics workers stopped")

    def _worker_loop(self) -> None:
        """Background worker loop for heavy computation."""
        while self._running:
            try:
                task = self._compute_queue.get(timeout=1.0)

                if task is None:  # Stop signal
                    break

                self._process_task(task)

            except Empty:
                continue
            except Exception as e:
                logger.exception(f"Worker error: {e}")

    def _process_task(self, task: Dict[str, Any]) -> None:
        """Process a computation task."""
        task_type = task.get("type")

        if task_type == "naturalness":
            self._compute_naturalness(task)
        elif task_type == "aggregate":
            self._compute_aggregation(task)
        else:
            logger.warning(f"Unknown task type: {task_type}")

    def _compute_naturalness(self, task: Dict[str, Any]) -> None:
        """Compute naturalness metrics for a synthesis."""
        synthesis_id = task["synthesis_id"]
        audio_data = task["audio_data"]
        sample_rate = task.get("sample_rate", 24000)

        try:
            audio_array = self._analyzer.bytes_to_array(audio_data, sample_rate)
            if audio_array is None:
                return

            naturalness = self._analyzer.estimate_naturalness(audio_array, sample_rate)

            # Update in storage (would need to extend storage for this)
            # For now, log the result
            logger.debug(
                f"Computed naturalness for {synthesis_id}: "
                f"MOS={naturalness.mos_estimate:.2f}"
            )

        except Exception as e:
            logger.error(f"Naturalness computation failed for {synthesis_id}: {e}")

    def _compute_aggregation(self, task: Dict[str, Any]) -> None:
        """Compute daily aggregation for a voice."""
        voice_id = task["voice_id"]
        date = task["date"]

        try:
            from datetime import datetime
            if isinstance(date, str):
                date = datetime.strptime(date, "%Y-%m-%d")

            self.storage.aggregate_voice_metrics(voice_id, date)
            logger.debug(f"Aggregated metrics for {voice_id} on {date}")

        except Exception as e:
            logger.error(f"Aggregation failed for {voice_id}: {e}")

    # ========================================================================
    # RECORDING
    # ========================================================================

    async def record_synthesis(
        self,
        analytics: SynthesisAnalytics,
        compute_heavy: bool = False,
        audio_data: Optional[bytes] = None,
    ) -> None:
        """
        Record synthesis analytics.

        Args:
            analytics: SynthesisAnalytics instance
            compute_heavy: Queue heavy computation (naturalness)
            audio_data: Raw audio for heavy computation
        """
        # Save to storage
        await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.storage.save_synthesis_metrics,
            analytics,
        )

        # Detect events
        if self.enable_event_detection:
            self._detect_events(analytics)

        # Queue heavy computation
        if compute_heavy and audio_data:
            self._compute_queue.put({
                "type": "naturalness",
                "synthesis_id": analytics.synthesis_id,
                "audio_data": audio_data,
                "sample_rate": 24000,
            })

        logger.debug(f"Recorded analytics: {analytics.synthesis_id}")

    def record_synthesis_sync(
        self,
        analytics: SynthesisAnalytics,
        compute_heavy: bool = False,
        audio_data: Optional[bytes] = None,
    ) -> None:
        """
        Record synthesis analytics (synchronous).

        Args:
            analytics: SynthesisAnalytics instance
            compute_heavy: Queue heavy computation
            audio_data: Raw audio for heavy computation
        """
        self.storage.save_synthesis_metrics(analytics)

        if self.enable_event_detection:
            self._detect_events(analytics)

        if compute_heavy and audio_data:
            self._compute_queue.put({
                "type": "naturalness",
                "synthesis_id": analytics.synthesis_id,
                "audio_data": audio_data,
            })

    # ========================================================================
    # SUMMARIES
    # ========================================================================

    async def get_voice_summary(
        self,
        voice_id: str,
        period_days: int = 7,
    ) -> VoiceAggregateMetrics:
        """
        Get aggregated metrics for a voice.

        Args:
            voice_id: Voice identifier
            period_days: Number of days to aggregate

        Returns:
            VoiceAggregateMetrics instance
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.storage.get_voice_summary,
            voice_id,
            period_days,
        )

    async def get_system_summary(
        self,
        period_days: int = 7,
    ) -> SystemAnalyticsSummary:
        """
        Get system-wide analytics summary.

        Args:
            period_days: Number of days to aggregate

        Returns:
            SystemAnalyticsSummary instance
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.storage.get_system_summary,
            period_days,
        )

    async def get_trends(
        self,
        metric: str,
        voice_id: Optional[str] = None,
        period_days: int = 30,
        granularity: str = "day",
    ) -> Dict[str, Any]:
        """
        Get time-series trend data for a metric.

        Args:
            metric: Metric name (quality, latency, naturalness)
            voice_id: Optional voice filter
            period_days: Number of days
            granularity: Time granularity (hour, day, week)

        Returns:
            Dictionary with trend data and analysis
        """
        data = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.storage.get_trend_data,
            metric,
            voice_id,
            period_days,
            granularity,
        )

        # Analyze trend
        trend_analysis = self._analyze_trend(data)

        return {
            "metric": metric,
            "voice_id": voice_id,
            "period_days": period_days,
            "granularity": granularity,
            "data": data,
            "trend": trend_analysis,
        }

    def _analyze_trend(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trend direction and detect anomalies."""
        if len(data) < 2:
            return {"direction": "insufficient_data", "change_percent": 0, "anomalies": []}

        values = [d["value"] for d in data if d["value"] is not None]
        if len(values) < 2:
            return {"direction": "insufficient_data", "change_percent": 0, "anomalies": []}

        # Simple trend: compare first half average to second half
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(values[mid:]) / (len(values) - mid) if len(values) > mid else 0

        if first_half_avg == 0:
            change_percent = 0
        else:
            change_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100

        # Determine direction
        if abs(change_percent) < 5:
            direction = "stable"
        elif change_percent > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Detect anomalies (values > 2 std deviations from mean)
        anomalies = []
        if len(values) > 5:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5

            for i, d in enumerate(data):
                if d["value"] is not None and abs(d["value"] - mean) > 2 * std:
                    anomalies.append({
                        "date": d["date"],
                        "value": d["value"],
                        "deviation": (d["value"] - mean) / std if std > 0 else 0,
                    })

        return {
            "direction": direction,
            "change_percent": round(change_percent, 2),
            "anomalies": anomalies[:5],  # Limit to 5
        }

    # ========================================================================
    # EVENT DETECTION
    # ========================================================================

    def _detect_events(self, analytics: SynthesisAnalytics) -> None:
        """Detect notable events from analytics."""
        events = []

        # Quality drop
        quality = analytics.get_quality_score()
        if quality < self._quality_threshold:
            events.append(AnalyticsEvent(
                event_type=AnalyticsEventType.QUALITY_DROP,
                timestamp=time.time(),
                voice_id=analytics.voice_id,
                synthesis_id=analytics.synthesis_id,
                severity=EventSeverity.WARNING if quality > 0.2 else EventSeverity.CRITICAL,
                message=f"Quality score dropped to {quality:.2f}",
                data={"quality_score": quality},
            ))

        # Latency spike
        if analytics.performance and analytics.performance.synthesis_latency_ms > self._latency_threshold_ms:
            events.append(AnalyticsEvent(
                event_type=AnalyticsEventType.LATENCY_SPIKE,
                timestamp=time.time(),
                voice_id=analytics.voice_id,
                synthesis_id=analytics.synthesis_id,
                severity=EventSeverity.WARNING,
                message=f"Latency spike: {analytics.performance.synthesis_latency_ms:.0f}ms",
                data={"latency_ms": analytics.performance.synthesis_latency_ms},
            ))

        # Artifact detection
        if analytics.technical_quality and analytics.technical_quality.artifact_score > self._artifact_threshold:
            events.append(AnalyticsEvent(
                event_type=AnalyticsEventType.ARTIFACT_DETECTED,
                timestamp=time.time(),
                voice_id=analytics.voice_id,
                synthesis_id=analytics.synthesis_id,
                severity=EventSeverity.WARNING,
                message=f"Artifacts detected: score={analytics.technical_quality.artifact_score:.2f}",
                data={"artifact_score": analytics.technical_quality.artifact_score},
            ))

        # Governance refusal
        if analytics.governance_action == "refuse":
            events.append(AnalyticsEvent(
                event_type=AnalyticsEventType.GOVERNANCE_REFUSAL,
                timestamp=time.time(),
                voice_id=analytics.voice_id,
                synthesis_id=analytics.synthesis_id,
                severity=EventSeverity.INFO,
                message="Synthesis refused by governance",
                data={"manipulation_detected": analytics.manipulation_detected},
            ))

        # Record events
        for event in events:
            try:
                self.storage.record_event(event)
            except Exception as e:
                logger.error(f"Failed to record event: {e}")

    async def get_recent_events(
        self,
        event_type: Optional[AnalyticsEventType] = None,
        voice_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AnalyticsEvent]:
        """
        Get recent analytics events.

        Args:
            event_type: Optional event type filter
            voice_id: Optional voice filter
            limit: Maximum events to return

        Returns:
            List of AnalyticsEvent instances
        """
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.storage.get_recent_events,
            event_type,
            voice_id,
            limit,
        )

    # ========================================================================
    # QUALITY DISTRIBUTION
    # ========================================================================

    async def get_quality_distribution(
        self,
        voice_id: Optional[str] = None,
        period_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get quality score distribution.

        Args:
            voice_id: Optional voice filter
            period_days: Number of days

        Returns:
            Quality distribution statistics
        """
        # Get trend data for quality
        quality_data = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.storage.get_trend_data,
            "quality",
            voice_id,
            period_days,
            "day",
        )

        if not quality_data:
            return {
                "voice_id": voice_id,
                "period_days": period_days,
                "distribution": {},
                "stats": {},
            }

        values = [d["value"] for d in quality_data if d["value"] is not None]

        # Compute distribution buckets
        buckets = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0}
        for v in values:
            if v >= 0.85:
                buckets["excellent"] += 1
            elif v >= 0.7:
                buckets["good"] += 1
            elif v >= 0.5:
                buckets["acceptable"] += 1
            else:
                buckets["poor"] += 1

        # Compute stats
        if values:
            mean = sum(values) / len(values)
            values_sorted = sorted(values)
            median = values_sorted[len(values) // 2]
            min_val = min(values)
            max_val = max(values)
        else:
            mean = median = min_val = max_val = 0

        return {
            "voice_id": voice_id,
            "period_days": period_days,
            "distribution": buckets,
            "stats": {
                "mean": round(mean, 3),
                "median": round(median, 3),
                "min": round(min_val, 3),
                "max": round(max_val, 3),
                "count": len(values),
            },
        }


# ============================================================================
# GLOBAL SERVICE INSTANCE
# ============================================================================

_default_service: Optional[VoxAnalyticsService] = None


def get_analytics_service() -> Optional[VoxAnalyticsService]:
    """Get the default analytics service (may be None if not initialized)."""
    return _default_service


def init_analytics_service(
    storage: Optional[AnalyticsStorage] = None,
    db_path: str = "axiom_vox_analytics.db",
    start_workers: bool = True,
) -> VoxAnalyticsService:
    """
    Initialize the default analytics service.

    Args:
        storage: Optional AnalyticsStorage instance
        db_path: Database path if storage not provided
        start_workers: Whether to start background workers

    Returns:
        VoxAnalyticsService instance
    """
    global _default_service

    if storage is None:
        storage = AnalyticsStorage(db_path=db_path)

    _default_service = VoxAnalyticsService(storage)

    if start_workers:
        _default_service.start_workers()

    return _default_service


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 70)
    print("  AXIOM VOX Analytics Service Demo")
    print("=" * 70)

    async def main():
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_analytics.db")
            storage = AnalyticsStorage(db_path=db_path)
            service = VoxAnalyticsService(storage)
            service.start_workers()

            print(f"\nDatabase: {db_path}")
            print("-" * 70)

            # Create and record sample analytics
            from axiom_vox.analytics.models import (
                TechnicalQualityMetrics,
                PerformanceMetrics,
                ComputationStatus,
            )

            for i in range(10):
                analytics = SynthesisAnalytics(
                    synthesis_id=f"synth_{i:03d}",
                    voice_id="demo_voice",
                    timestamp=time.time() - (i * 3600),
                    text_length=100 + i * 10,
                    technical_quality=TechnicalQualityMetrics(
                        snr_db=25 + i % 5,
                        peak_amplitude=0.8,
                        rms_level_db=-18.0,
                        dynamic_range_db=12.0,
                        silence_ratio=0.1,
                        clipping_samples=0,
                        artifact_score=0.1 if i < 8 else 0.5,  # Trigger artifact event
                    ),
                    performance=PerformanceMetrics(
                        synthesis_latency_ms=200 + i * 50,
                        audio_duration_ms=2000,
                        real_time_factor=0.1,
                        characters_per_second=400,
                    ),
                    computation_status=ComputationStatus.COMPLETE,
                )
                await service.record_synthesis(analytics)
                print(f"Recorded: {analytics.synthesis_id}")

            # Wait a bit for event detection
            await asyncio.sleep(0.5)

            # Get system summary
            summary = await service.get_system_summary(period_days=1)
            print(f"\nSystem Summary:")
            print(f"  Total Syntheses: {summary.total_syntheses}")
            print(f"  P95 Latency: {summary.p95_latency_ms:.1f}ms")

            # Get voice summary
            voice_summary = await service.get_voice_summary("demo_voice", period_days=1)
            print(f"\nVoice Summary (demo_voice):")
            print(f"  Total Syntheses: {voice_summary.total_syntheses}")
            print(f"  Avg Quality: {voice_summary.avg_quality_score:.2f}")

            # Get trends
            trends = await service.get_trends("latency", period_days=1)
            print(f"\nLatency Trend: {trends['trend']['direction']}")

            # Get quality distribution
            distribution = await service.get_quality_distribution(period_days=1)
            print(f"\nQuality Distribution: {distribution['distribution']}")

            # Get events
            events = await service.get_recent_events(limit=5)
            print(f"\nRecent Events: {len(events)}")
            for event in events[:3]:
                print(f"  - {event.event_type.value}: {event.message}")

            service.stop_workers()

    asyncio.run(main())

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
