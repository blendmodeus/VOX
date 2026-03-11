"""
AXIOM VOX Analytics Storage
---------------------------

Database storage layer for voice analytics.

Uses SQLite following the VoxDatabase pattern from persistence.py.

Tables:
- synthesis_metrics: Per-synthesis analytics records
- voice_metrics_daily: Daily aggregates per voice
- analytics_events: Notable events for trend analysis

Usage:
    from axiom_vox.analytics import AnalyticsStorage

    storage = AnalyticsStorage(db_path="analytics.db")
    storage.save_synthesis_metrics(analytics)
    summary = storage.get_voice_summary("voice_id", days=7)
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from axiom_vox.analytics.models import (
    SynthesisAnalytics,
    StreamingSessionAnalytics,
    TechnicalQualityMetrics,
    SpectralQualityMetrics,
    NaturalnessMetrics,
    PerformanceMetrics,
    VoiceAggregateMetrics,
    SystemAnalyticsSummary,
    AnalyticsEvent,
    AnalyticsEventType,
    EventSeverity,
    ComputationStatus,
    QualityTier,
)

logger = logging.getLogger(__name__)


class AnalyticsStorage:
    """
    SQLite storage for voice analytics.

    Thread-safe with connection pooling per thread.
    """

    def __init__(self, db_path: str = "axiom_vox_analytics.db"):
        """
        Initialize analytics storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Per-synthesis metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS synthesis_metrics (
                synthesis_id TEXT PRIMARY KEY,
                voice_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                text_length INTEGER,
                audio_duration_ms REAL,

                -- Technical quality
                snr_db REAL,
                peak_amplitude REAL,
                rms_level_db REAL,
                dynamic_range_db REAL,
                silence_ratio REAL,
                clipping_samples INTEGER,
                artifact_score REAL,

                -- Spectral quality
                spectral_centroid_hz REAL,
                spectral_bandwidth_hz REAL,
                spectral_rolloff_hz REAL,
                spectral_flatness REAL,
                harmonic_ratio REAL,

                -- Naturalness
                mos_estimate REAL,
                naturalness_score REAL,
                prosody_score REAL,
                articulation_score REAL,

                -- Performance
                synthesis_latency_ms REAL,
                first_chunk_latency_ms REAL,
                real_time_factor REAL,
                characters_per_second REAL,

                -- Governance
                governance_action TEXT,
                content_emotion_match REAL,
                manipulation_detected INTEGER DEFAULT 0,

                -- Composite scores
                quality_score REAL,
                quality_tier TEXT,

                -- Status
                computation_status TEXT DEFAULT 'pending',
                error_message TEXT,

                -- Full JSON for extensibility
                full_metrics TEXT,

                -- Timestamps
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Daily voice aggregates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_metrics_daily (
                voice_id TEXT NOT NULL,
                date TEXT NOT NULL,

                -- Volume
                total_syntheses INTEGER DEFAULT 0,
                total_audio_seconds REAL DEFAULT 0,
                total_characters INTEGER DEFAULT 0,

                -- Quality averages
                avg_mos_estimate REAL,
                avg_naturalness REAL,
                avg_snr_db REAL,
                avg_quality_score REAL,
                avg_reference_similarity REAL,

                -- Performance percentiles
                p50_latency_ms REAL,
                p95_latency_ms REAL,
                p99_latency_ms REAL,
                avg_real_time_factor REAL,

                -- Quality distribution (JSON)
                quality_distribution TEXT,

                -- Issues
                error_count INTEGER DEFAULT 0,
                governance_refusals INTEGER DEFAULT 0,
                artifact_detections INTEGER DEFAULT 0,

                -- Timestamps
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (voice_id, date)
            )
        """)

        # Analytics events for trends
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                voice_id TEXT,
                synthesis_id TEXT,
                severity TEXT DEFAULT 'info',
                message TEXT,
                data TEXT,
                resolved INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Streaming session metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS streaming_sessions (
                session_id TEXT PRIMARY KEY,
                voice_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                text_length INTEGER,

                -- Chunk statistics
                total_chunks INTEGER DEFAULT 0,
                total_bytes INTEGER DEFAULT 0,
                total_audio_duration_ms REAL DEFAULT 0,
                sentences_completed INTEGER DEFAULT 0,

                -- Latency metrics
                first_chunk_latency_ms REAL,
                avg_chunk_latency_ms REAL,
                latency_std_ms REAL,
                latency_min_ms REAL,
                latency_max_ms REAL,
                late_chunk_ratio REAL DEFAULT 0,
                late_chunks INTEGER DEFAULT 0,

                -- Quality metrics
                stutters_detected INTEGER DEFAULT 0,
                quality_drops INTEGER DEFAULT 0,
                avg_stream_quality REAL DEFAULT 1.0,
                streaming_rtf REAL,

                -- Link to synthesis metrics
                synthesis_id TEXT,

                -- Timestamps
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (synthesis_id) REFERENCES synthesis_metrics(synthesis_id)
            )
        """)

        # Streaming events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS streaming_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                chunk_index INTEGER,
                severity TEXT DEFAULT 'info',
                message TEXT,
                data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (session_id) REFERENCES streaming_sessions(session_id)
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_synth_voice ON synthesis_metrics(voice_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_synth_ts ON synthesis_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_synth_date ON synthesis_metrics(date(datetime(timestamp, 'unixepoch')))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON analytics_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON analytics_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stream_voice ON streaming_sessions(voice_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stream_ts ON streaming_sessions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stream_events ON streaming_events(session_id)")

        conn.commit()
        logger.info(f"Analytics database initialized: {self.db_path}")

    # ========================================================================
    # SYNTHESIS METRICS
    # ========================================================================

    def save_synthesis_metrics(self, analytics: SynthesisAnalytics) -> None:
        """
        Save synthesis analytics to database.

        Args:
            analytics: SynthesisAnalytics instance to save
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Extract nested metrics
        tech = analytics.technical_quality
        spectral = analytics.spectral_quality
        naturalness = analytics.naturalness
        perf = analytics.performance

        cursor.execute("""
            INSERT OR REPLACE INTO synthesis_metrics (
                synthesis_id, voice_id, timestamp, text_length, audio_duration_ms,
                snr_db, peak_amplitude, rms_level_db, dynamic_range_db,
                silence_ratio, clipping_samples, artifact_score,
                spectral_centroid_hz, spectral_bandwidth_hz, spectral_rolloff_hz,
                spectral_flatness, harmonic_ratio,
                mos_estimate, naturalness_score, prosody_score, articulation_score,
                synthesis_latency_ms, first_chunk_latency_ms, real_time_factor,
                characters_per_second,
                governance_action, content_emotion_match, manipulation_detected,
                quality_score, quality_tier, computation_status, full_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analytics.synthesis_id,
            analytics.voice_id,
            analytics.timestamp,
            analytics.text_length,
            perf.audio_duration_ms if perf else None,
            tech.snr_db if tech else None,
            tech.peak_amplitude if tech else None,
            tech.rms_level_db if tech else None,
            tech.dynamic_range_db if tech else None,
            tech.silence_ratio if tech else None,
            tech.clipping_samples if tech else None,
            tech.artifact_score if tech else None,
            spectral.spectral_centroid_hz if spectral else None,
            spectral.spectral_bandwidth_hz if spectral else None,
            spectral.spectral_rolloff_hz if spectral else None,
            spectral.spectral_flatness if spectral else None,
            spectral.harmonic_ratio if spectral else None,
            naturalness.mos_estimate if naturalness else None,
            naturalness.overall_naturalness if naturalness else None,
            naturalness.prosody_score if naturalness else None,
            naturalness.articulation_score if naturalness else None,
            perf.synthesis_latency_ms if perf else None,
            perf.first_chunk_latency_ms if perf else None,
            perf.real_time_factor if perf else None,
            perf.characters_per_second if perf else None,
            analytics.governance_action,
            analytics.content_emotion_match,
            1 if analytics.manipulation_detected else 0,
            analytics.get_quality_score(),
            analytics.get_quality_tier().value,
            analytics.computation_status.value,
            analytics.to_json(),
        ))

        conn.commit()
        logger.debug(f"Saved synthesis metrics: {analytics.synthesis_id}")

    def get_synthesis_metrics(self, synthesis_id: str) -> Optional[SynthesisAnalytics]:
        """
        Retrieve synthesis analytics by ID.

        Args:
            synthesis_id: Synthesis identifier

        Returns:
            SynthesisAnalytics instance, or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT full_metrics FROM synthesis_metrics WHERE synthesis_id = ?",
            (synthesis_id,),
        )
        row = cursor.fetchone()

        if not row:
            return None

        return SynthesisAnalytics.from_json(row["full_metrics"])

    def update_computation_status(
        self,
        synthesis_id: str,
        status: ComputationStatus,
        error: Optional[str] = None,
    ) -> None:
        """Update computation status for a synthesis."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE synthesis_metrics SET computation_status = ?, error_message = ? WHERE synthesis_id = ?",
            (status.value, error, synthesis_id),
        )
        conn.commit()

    # ========================================================================
    # AGGREGATION
    # ========================================================================

    def aggregate_voice_metrics(
        self,
        voice_id: str,
        date: datetime,
    ) -> VoiceAggregateMetrics:
        """
        Compute and store daily voice aggregates.

        Args:
            voice_id: Voice identifier
            date: Date to aggregate

        Returns:
            VoiceAggregateMetrics instance
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        date_str = date.strftime("%Y-%m-%d")
        start_ts = datetime(date.year, date.month, date.day).timestamp()
        end_ts = start_ts + 86400  # 24 hours

        # Aggregate metrics
        cursor.execute("""
            SELECT
                COUNT(*) as total_syntheses,
                SUM(audio_duration_ms) / 1000.0 as total_audio_seconds,
                SUM(text_length) as total_characters,
                AVG(mos_estimate) as avg_mos_estimate,
                AVG(naturalness_score) as avg_naturalness,
                AVG(snr_db) as avg_snr_db,
                AVG(quality_score) as avg_quality_score,
                SUM(CASE WHEN governance_action = 'refuse' THEN 1 ELSE 0 END) as governance_refusals,
                SUM(CASE WHEN artifact_score > 0.3 THEN 1 ELSE 0 END) as artifact_detections,
                SUM(CASE WHEN computation_status = 'failed' THEN 1 ELSE 0 END) as error_count
            FROM synthesis_metrics
            WHERE voice_id = ? AND timestamp >= ? AND timestamp < ?
        """, (voice_id, start_ts, end_ts))

        row = cursor.fetchone()

        # Compute latency percentiles
        cursor.execute("""
            SELECT synthesis_latency_ms
            FROM synthesis_metrics
            WHERE voice_id = ? AND timestamp >= ? AND timestamp < ?
                AND synthesis_latency_ms IS NOT NULL
            ORDER BY synthesis_latency_ms
        """, (voice_id, start_ts, end_ts))

        latencies = [r["synthesis_latency_ms"] for r in cursor.fetchall()]
        p50, p95, p99 = self._compute_percentiles(latencies, [50, 95, 99])

        # Compute quality distribution
        cursor.execute("""
            SELECT quality_tier, COUNT(*) as count
            FROM synthesis_metrics
            WHERE voice_id = ? AND timestamp >= ? AND timestamp < ?
            GROUP BY quality_tier
        """, (voice_id, start_ts, end_ts))

        quality_distribution = {r["quality_tier"]: r["count"] for r in cursor.fetchall()}

        # Compute avg RTF
        cursor.execute("""
            SELECT AVG(real_time_factor) as avg_rtf
            FROM synthesis_metrics
            WHERE voice_id = ? AND timestamp >= ? AND timestamp < ?
                AND real_time_factor IS NOT NULL
        """, (voice_id, start_ts, end_ts))
        avg_rtf = cursor.fetchone()["avg_rtf"] or 0.0

        metrics = VoiceAggregateMetrics(
            voice_id=voice_id,
            period_start=datetime.fromtimestamp(start_ts),
            period_end=datetime.fromtimestamp(end_ts),
            total_syntheses=row["total_syntheses"] or 0,
            total_audio_seconds=row["total_audio_seconds"] or 0.0,
            total_characters=row["total_characters"] or 0,
            avg_mos_estimate=row["avg_mos_estimate"] or 0.0,
            avg_naturalness=row["avg_naturalness"] or 0.0,
            avg_snr_db=row["avg_snr_db"] or 0.0,
            avg_quality_score=row["avg_quality_score"] or 0.0,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            avg_real_time_factor=avg_rtf,
            quality_distribution=quality_distribution,
            error_count=row["error_count"] or 0,
            governance_refusals=row["governance_refusals"] or 0,
            artifact_detections=row["artifact_detections"] or 0,
        )

        # Save to daily table
        cursor.execute("""
            INSERT OR REPLACE INTO voice_metrics_daily (
                voice_id, date, total_syntheses, total_audio_seconds, total_characters,
                avg_mos_estimate, avg_naturalness, avg_snr_db, avg_quality_score,
                p50_latency_ms, p95_latency_ms, p99_latency_ms, avg_real_time_factor,
                quality_distribution, error_count, governance_refusals, artifact_detections,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            voice_id, date_str,
            metrics.total_syntheses, metrics.total_audio_seconds, metrics.total_characters,
            metrics.avg_mos_estimate, metrics.avg_naturalness, metrics.avg_snr_db,
            metrics.avg_quality_score, metrics.p50_latency_ms, metrics.p95_latency_ms,
            metrics.p99_latency_ms, metrics.avg_real_time_factor,
            json.dumps(quality_distribution),
            metrics.error_count, metrics.governance_refusals, metrics.artifact_detections,
        ))

        conn.commit()
        return metrics

    def get_voice_summary(
        self,
        voice_id: str,
        period_days: int = 7,
    ) -> VoiceAggregateMetrics:
        """
        Get aggregated metrics for a voice over a period.

        Args:
            voice_id: Voice identifier
            period_days: Number of days to aggregate

        Returns:
            VoiceAggregateMetrics instance
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        cursor.execute("""
            SELECT
                SUM(total_syntheses) as total_syntheses,
                SUM(total_audio_seconds) as total_audio_seconds,
                SUM(total_characters) as total_characters,
                AVG(avg_mos_estimate) as avg_mos_estimate,
                AVG(avg_naturalness) as avg_naturalness,
                AVG(avg_snr_db) as avg_snr_db,
                AVG(avg_quality_score) as avg_quality_score,
                AVG(p50_latency_ms) as p50_latency_ms,
                AVG(p95_latency_ms) as p95_latency_ms,
                AVG(p99_latency_ms) as p99_latency_ms,
                AVG(avg_real_time_factor) as avg_rtf,
                SUM(error_count) as error_count,
                SUM(governance_refusals) as governance_refusals,
                SUM(artifact_detections) as artifact_detections
            FROM voice_metrics_daily
            WHERE voice_id = ? AND date >= ? AND date <= ?
        """, (voice_id, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))

        row = cursor.fetchone()

        return VoiceAggregateMetrics(
            voice_id=voice_id,
            period_start=start_date,
            period_end=end_date,
            total_syntheses=row["total_syntheses"] or 0,
            total_audio_seconds=row["total_audio_seconds"] or 0.0,
            total_characters=row["total_characters"] or 0,
            avg_mos_estimate=row["avg_mos_estimate"] or 0.0,
            avg_naturalness=row["avg_naturalness"] or 0.0,
            avg_snr_db=row["avg_snr_db"] or 0.0,
            avg_quality_score=row["avg_quality_score"] or 0.0,
            p50_latency_ms=row["p50_latency_ms"] or 0.0,
            p95_latency_ms=row["p95_latency_ms"] or 0.0,
            p99_latency_ms=row["p99_latency_ms"] or 0.0,
            avg_real_time_factor=row["avg_rtf"] or 0.0,
            error_count=row["error_count"] or 0,
            governance_refusals=row["governance_refusals"] or 0,
            artifact_detections=row["artifact_detections"] or 0,
        )

    def get_system_summary(self, period_days: int = 7) -> SystemAnalyticsSummary:
        """
        Get system-wide analytics summary.

        Args:
            period_days: Number of days to aggregate

        Returns:
            SystemAnalyticsSummary instance
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()

        # Aggregate across all voices
        cursor.execute("""
            SELECT
                COUNT(*) as total_syntheses,
                SUM(audio_duration_ms) / 1000.0 / 3600.0 as total_audio_hours,
                COUNT(DISTINCT voice_id) as unique_voices,
                AVG(mos_estimate) as avg_mos_estimate,
                AVG(naturalness_score) as avg_naturalness,
                AVG(snr_db) as avg_snr_db,
                AVG(synthesis_latency_ms) as avg_latency_ms,
                SUM(CASE WHEN computation_status = 'failed' THEN 1 ELSE 0 END) as total_errors,
                SUM(CASE WHEN governance_action = 'refuse' THEN 1 ELSE 0 END) as governance_refusals,
                SUM(CASE WHEN artifact_score > 0.3 THEN 1 ELSE 0 END) as artifact_detections
            FROM synthesis_metrics
            WHERE timestamp >= ? AND timestamp < ?
        """, (start_ts, end_ts))

        row = cursor.fetchone()
        total_syntheses = row["total_syntheses"] or 0

        # Compute latency percentiles
        cursor.execute("""
            SELECT synthesis_latency_ms
            FROM synthesis_metrics
            WHERE timestamp >= ? AND timestamp < ? AND synthesis_latency_ms IS NOT NULL
            ORDER BY synthesis_latency_ms
        """, (start_ts, end_ts))

        latencies = [r["synthesis_latency_ms"] for r in cursor.fetchall()]
        p50, p95, p99 = self._compute_percentiles(latencies, [50, 95, 99])

        # Quality percentiles
        cursor.execute("""
            SELECT quality_score
            FROM synthesis_metrics
            WHERE timestamp >= ? AND timestamp < ? AND quality_score IS NOT NULL
            ORDER BY quality_score
        """, (start_ts, end_ts))

        qualities = [r["quality_score"] for r in cursor.fetchall()]
        q50, q95 = self._compute_percentiles(qualities, [50, 95])

        # Top voices
        cursor.execute("""
            SELECT voice_id, COUNT(*) as count, AVG(quality_score) as avg_quality
            FROM synthesis_metrics
            WHERE timestamp >= ? AND timestamp < ?
            GROUP BY voice_id
            ORDER BY count DESC
            LIMIT 5
        """, (start_ts, end_ts))

        top_voices = [
            {"voice_id": r["voice_id"], "syntheses": r["count"], "avg_quality": r["avg_quality"]}
            for r in cursor.fetchall()
        ]

        # Compute avg RTF
        cursor.execute("""
            SELECT AVG(real_time_factor) as avg_rtf
            FROM synthesis_metrics
            WHERE timestamp >= ? AND timestamp < ? AND real_time_factor IS NOT NULL
        """, (start_ts, end_ts))
        avg_rtf = cursor.fetchone()["avg_rtf"] or 0.0

        error_rate = (row["total_errors"] or 0) / total_syntheses if total_syntheses > 0 else 0

        return SystemAnalyticsSummary(
            period_start=start_date,
            period_end=end_date,
            total_syntheses=total_syntheses,
            total_audio_hours=row["total_audio_hours"] or 0.0,
            unique_voices=row["unique_voices"] or 0,
            avg_mos_estimate=row["avg_mos_estimate"] or 0.0,
            avg_naturalness=row["avg_naturalness"] or 0.0,
            avg_snr_db=row["avg_snr_db"] or 0.0,
            quality_score_p50=q50,
            quality_score_p95=q95,
            avg_latency_ms=row["avg_latency_ms"] or 0.0,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            avg_rtf=avg_rtf,
            total_errors=row["total_errors"] or 0,
            error_rate=error_rate,
            governance_refusals=row["governance_refusals"] or 0,
            artifact_detections=row["artifact_detections"] or 0,
            top_voices=top_voices,
        )

    # ========================================================================
    # TRENDS
    # ========================================================================

    def get_trend_data(
        self,
        metric: str,
        voice_id: Optional[str] = None,
        period_days: int = 30,
        granularity: str = "day",
    ) -> List[Dict[str, Any]]:
        """
        Get time-series trend data for a metric.

        Args:
            metric: Metric name (quality, latency, volume, naturalness)
            voice_id: Optional voice filter
            period_days: Number of days to query
            granularity: Time granularity (hour, day, week)

        Returns:
            List of {date, value, count} dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()

        # Map metric to column
        metric_columns = {
            "quality": "quality_score",
            "latency": "synthesis_latency_ms",
            "naturalness": "naturalness_score",
            "mos": "mos_estimate",
            "snr": "snr_db",
        }

        column = metric_columns.get(metric, "quality_score")

        # Date grouping based on granularity
        if granularity == "hour":
            date_format = "%Y-%m-%d %H:00"
            group_expr = "strftime('%Y-%m-%d %H:00', datetime(timestamp, 'unixepoch'))"
        elif granularity == "week":
            date_format = "%Y-W%W"
            group_expr = "strftime('%Y-W%W', datetime(timestamp, 'unixepoch'))"
        else:  # day
            date_format = "%Y-%m-%d"
            group_expr = "date(datetime(timestamp, 'unixepoch'))"

        # Build query
        query = f"""
            SELECT
                {group_expr} as period,
                AVG({column}) as value,
                COUNT(*) as count
            FROM synthesis_metrics
            WHERE timestamp >= ? AND timestamp < ?
                AND {column} IS NOT NULL
        """
        params = [start_ts, end_ts]

        if voice_id:
            query += " AND voice_id = ?"
            params.append(voice_id)

        query += f" GROUP BY {group_expr} ORDER BY period"

        cursor.execute(query, params)

        return [
            {"date": row["period"], "value": row["value"], "count": row["count"]}
            for row in cursor.fetchall()
        ]

    # ========================================================================
    # EVENTS
    # ========================================================================

    def record_event(self, event: AnalyticsEvent) -> int:
        """
        Record an analytics event.

        Args:
            event: AnalyticsEvent instance

        Returns:
            Event ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO analytics_events (
                timestamp, event_type, voice_id, synthesis_id,
                severity, message, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp,
            event.event_type.value,
            event.voice_id,
            event.synthesis_id,
            event.severity.value,
            event.message,
            json.dumps(event.data),
        ))

        conn.commit()
        return cursor.lastrowid

    def get_recent_events(
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
            limit: Maximum number of events to return

        Returns:
            List of AnalyticsEvent instances
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM analytics_events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)

        if voice_id:
            query += " AND voice_id = ?"
            params.append(voice_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        events = []
        for row in cursor.fetchall():
            events.append(AnalyticsEvent(
                event_type=AnalyticsEventType(row["event_type"]),
                timestamp=row["timestamp"],
                voice_id=row["voice_id"],
                synthesis_id=row["synthesis_id"],
                severity=EventSeverity(row["severity"]),
                message=row["message"],
                data=json.loads(row["data"]) if row["data"] else {},
                resolved=bool(row["resolved"]),
            ))

        return events

    # ========================================================================
    # STREAMING SESSIONS
    # ========================================================================

    def save_streaming_session(
        self,
        session: "StreamingSessionAnalytics",
    ) -> None:
        """
        Save streaming session analytics to database.

        Args:
            session: StreamingSessionAnalytics instance to save
        """
        from axiom_vox.analytics.models import StreamingSessionAnalytics

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO streaming_sessions (
                session_id, voice_id, timestamp, text_length,
                total_chunks, total_bytes, total_audio_duration_ms, sentences_completed,
                first_chunk_latency_ms, avg_chunk_latency_ms, latency_std_ms,
                latency_min_ms, latency_max_ms, late_chunk_ratio, late_chunks,
                stutters_detected, quality_drops, avg_stream_quality, streaming_rtf,
                synthesis_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.session_id,
            session.voice_id,
            session.timestamp,
            session.text_length,
            session.total_chunks,
            session.total_bytes,
            session.total_audio_duration_ms,
            session.sentences_completed,
            session.first_chunk_latency_ms,
            session.avg_chunk_latency_ms,
            session.latency_std_ms,
            session.latency_min_ms,
            session.latency_max_ms,
            session.late_chunk_ratio,
            session.late_chunks,
            session.stutters_detected,
            session.quality_drops,
            session.avg_stream_quality,
            session.streaming_rtf,
            session.synthesis_analytics.synthesis_id if session.synthesis_analytics else None,
        ))

        conn.commit()
        logger.debug(f"Saved streaming session: {session.session_id}")

    def save_streaming_event(self, event: Dict[str, Any]) -> int:
        """
        Save a streaming event.

        Args:
            event: Event dictionary with session_id, timestamp, event_type, etc.

        Returns:
            Event ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO streaming_events (
                session_id, timestamp, event_type, chunk_index,
                severity, message, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event["session_id"],
            event["timestamp"],
            event["event_type"],
            event.get("chunk_index"),
            event.get("severity", "info"),
            event.get("message", ""),
            json.dumps(event.get("data", {})),
        ))

        conn.commit()
        return cursor.lastrowid

    def get_streaming_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get streaming session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary, or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM streaming_sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()

        if not row:
            return None

        return dict(row)

    def get_streaming_summary(
        self,
        voice_id: Optional[str] = None,
        period_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get streaming performance summary.

        Args:
            voice_id: Optional voice filter
            period_days: Number of days to aggregate

        Returns:
            Summary dictionary with streaming metrics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        start_ts = start_date.timestamp()

        query = """
            SELECT
                COUNT(*) as total_sessions,
                AVG(first_chunk_latency_ms) as avg_first_chunk_ms,
                AVG(avg_chunk_latency_ms) as avg_chunk_latency_ms,
                AVG(late_chunk_ratio) as avg_late_ratio,
                SUM(stutters_detected) as total_stutters,
                SUM(quality_drops) as total_quality_drops,
                AVG(streaming_rtf) as avg_rtf,
                SUM(total_chunks) as total_chunks,
                SUM(total_bytes) as total_bytes,
                SUM(total_audio_duration_ms) / 1000.0 / 3600.0 as total_audio_hours
            FROM streaming_sessions
            WHERE timestamp >= ?
        """
        params = [start_ts]

        if voice_id:
            query += " AND voice_id = ?"
            params.append(voice_id)

        cursor.execute(query, params)
        row = cursor.fetchone()

        return {
            "period_days": period_days,
            "voice_id": voice_id,
            "total_sessions": row["total_sessions"] or 0,
            "total_chunks": row["total_chunks"] or 0,
            "total_bytes": row["total_bytes"] or 0,
            "total_audio_hours": row["total_audio_hours"] or 0.0,
            "avg_first_chunk_ms": row["avg_first_chunk_ms"],
            "avg_chunk_latency_ms": row["avg_chunk_latency_ms"],
            "avg_late_ratio": row["avg_late_ratio"],
            "total_stutters": row["total_stutters"] or 0,
            "total_quality_drops": row["total_quality_drops"] or 0,
            "avg_rtf": row["avg_rtf"],
        }

    def get_streaming_events(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get streaming events.

        Args:
            session_id: Optional session filter
            event_type: Optional event type filter
            limit: Maximum events to return

        Returns:
            List of event dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM streaming_events WHERE 1=1"
        params = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "chunk_index": row["chunk_index"],
                "severity": row["severity"],
                "message": row["message"],
                "data": json.loads(row["data"]) if row["data"] else {},
            }
            for row in cursor.fetchall()
        ]

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _compute_percentiles(
        self,
        values: List[float],
        percentiles: List[int],
    ) -> List[float]:
        """Compute percentiles from a sorted list of values."""
        if not values:
            return [0.0] * len(percentiles)

        results = []
        n = len(values)
        for p in percentiles:
            idx = int(n * p / 100)
            idx = min(idx, n - 1)
            results.append(values[idx])

        return results

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import os
    import time

    print("=" * 70)
    print("  AXIOM VOX Analytics Storage Demo")
    print("=" * 70)

    # Create temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_analytics.db")
        storage = AnalyticsStorage(db_path=db_path)

        print(f"\nDatabase: {db_path}")
        print("-" * 70)

        # Create sample analytics
        from axiom_vox.analytics.models import (
            SynthesisAnalytics,
            TechnicalQualityMetrics,
            PerformanceMetrics,
            ComputationStatus,
        )

        for i in range(5):
            analytics = SynthesisAnalytics(
                synthesis_id=f"synth_{i:03d}",
                voice_id="test_voice",
                timestamp=time.time() - (i * 3600),  # Spread over hours
                text_length=100 + i * 10,
                technical_quality=TechnicalQualityMetrics(
                    snr_db=25 + i,
                    peak_amplitude=0.8,
                    rms_level_db=-18.0,
                    dynamic_range_db=12.0,
                    silence_ratio=0.1,
                    clipping_samples=0,
                    artifact_score=0.1,
                ),
                performance=PerformanceMetrics(
                    synthesis_latency_ms=200 + i * 50,
                    audio_duration_ms=2000 + i * 100,
                    real_time_factor=0.1,
                    characters_per_second=400,
                ),
                computation_status=ComputationStatus.COMPLETE,
            )
            storage.save_synthesis_metrics(analytics)
            print(f"Saved: {analytics.synthesis_id}")

        # Retrieve one
        retrieved = storage.get_synthesis_metrics("synth_002")
        if retrieved:
            print(f"\nRetrieved synth_002: quality={retrieved.get_quality_score():.2f}")

        # Get system summary
        summary = storage.get_system_summary(period_days=1)
        print(f"\nSystem Summary:")
        print(f"  Total Syntheses: {summary.total_syntheses}")
        print(f"  Avg Latency: {summary.avg_latency_ms:.1f}ms")
        print(f"  P95 Latency: {summary.p95_latency_ms:.1f}ms")

        # Get trends
        trends = storage.get_trend_data("latency", period_days=1, granularity="hour")
        print(f"\nLatency Trends: {len(trends)} data points")

        storage.close()

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
