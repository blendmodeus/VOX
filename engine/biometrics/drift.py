"""
Voice Drift Monitor
-------------------

Monitor voice biometric drift over time.

Detects:
    - Short-term drift (session-to-session changes)
    - Long-term drift (gradual voice changes over months/years)
    - Anomalous sudden changes (potential account compromise)

AXIØM Phase 5: Resonance - "finding signature frequency"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

import numpy as np

from .models import (
    DriftReport,
    DriftSeverity,
    DRIFT_SHORT_TERM_THRESHOLD,
    DRIFT_LONG_TERM_THRESHOLD,
)
from .embeddings import deserialize_embedding

logger = logging.getLogger(__name__)


@dataclass
class DriftSample:
    """A single drift measurement sample."""
    timestamp: float
    similarity: float
    context: str = "verification"


@dataclass
class DriftTrend:
    """Trend analysis results."""
    direction: str  # "stable", "increasing", "decreasing"
    slope: float
    confidence: float
    samples_analyzed: int


class DriftMonitor:
    """
    Monitor voice biometric drift over time.

    Tracks:
        - Rolling average similarity scores
        - Short-term volatility
        - Long-term trend direction
        - Anomaly detection
    """

    def __init__(
        self,
        short_term_threshold: float = DRIFT_SHORT_TERM_THRESHOLD,
        long_term_threshold: float = DRIFT_LONG_TERM_THRESHOLD,
        anomaly_threshold: float = 0.30,
    ):
        """
        Initialize drift monitor.

        Args:
            short_term_threshold: Threshold for short-term drift warning (0.15)
            long_term_threshold: Threshold for long-term drift requiring re-enrollment (0.25)
            anomaly_threshold: Threshold for sudden change anomaly detection (0.30)
        """
        self.short_term_threshold = short_term_threshold
        self.long_term_threshold = long_term_threshold
        self.anomaly_threshold = anomaly_threshold

    def analyze(
        self,
        voice_id: str,
        history: List[Dict[str, Any]],
        template_embedding: Optional[bytes] = None,
    ) -> DriftReport:
        """
        Analyze drift from embedding history.

        Args:
            voice_id: Voice ID
            history: List of embedding history entries from storage
            template_embedding: Current template embedding (for comparison)

        Returns:
            DriftReport with analysis
        """
        if len(history) < 2:
            return DriftReport(
                voice_id=voice_id,
                severity=DriftSeverity.NONE,
                message="Insufficient history for drift analysis",
                sample_count=len(history),
            )

        # Extract similarity scores with timestamps
        samples = []
        for entry in history:
            if entry.get("similarity_to_template") is not None:
                samples.append(DriftSample(
                    timestamp=entry["timestamp"],
                    similarity=entry["similarity_to_template"],
                    context=entry.get("context", "verification"),
                ))

        if len(samples) < 2:
            return DriftReport(
                voice_id=voice_id,
                severity=DriftSeverity.NONE,
                message="Insufficient similarity data for drift analysis",
                sample_count=len(history),
            )

        # Sort by timestamp
        samples.sort(key=lambda x: x.timestamp)

        # Calculate metrics
        similarities = np.array([s.similarity for s in samples])
        timestamps = np.array([s.timestamp for s in samples])

        # Short-term drift (recent samples vs baseline)
        short_term_drift = self._calculate_short_term_drift(similarities)

        # Long-term drift (trend over all samples)
        long_term_drift, trend = self._calculate_long_term_drift(similarities, timestamps)

        # Anomaly detection
        anomaly_detected, anomaly_details = self._detect_anomaly(similarities)

        # Determine severity
        severity = self._determine_severity(
            short_term_drift,
            long_term_drift,
            anomaly_detected,
        )

        # Recommendations
        requires_re_enrollment = (
            long_term_drift > self.long_term_threshold or
            (anomaly_detected and short_term_drift > self.anomaly_threshold)
        )
        update_recommended = (
            short_term_drift > self.short_term_threshold and
            not requires_re_enrollment
        )

        # Generate message
        message = self._generate_message(
            severity,
            short_term_drift,
            long_term_drift,
            anomaly_detected,
            requires_re_enrollment,
            update_recommended,
        )

        return DriftReport(
            voice_id=voice_id,
            severity=severity,
            short_term_drift=short_term_drift,
            long_term_drift=long_term_drift,
            trend_direction=trend.direction,
            short_term_threshold=self.short_term_threshold,
            long_term_threshold=self.long_term_threshold,
            requires_re_enrollment=requires_re_enrollment,
            update_recommended=update_recommended,
            sample_count=len(samples),
            first_sample_date=samples[0].timestamp,
            last_sample_date=samples[-1].timestamp,
            anomaly_detected=anomaly_detected,
            anomaly_details=anomaly_details,
            message=message,
        )

    def _calculate_short_term_drift(
        self,
        similarities: np.ndarray,
        recent_window: int = 5,
        baseline_window: int = 20,
    ) -> float:
        """
        Calculate short-term drift.

        Compares recent samples to baseline average.
        """
        if len(similarities) < recent_window:
            recent = similarities
            baseline = similarities
        else:
            recent = similarities[-recent_window:]
            if len(similarities) >= baseline_window + recent_window:
                baseline = similarities[-(baseline_window + recent_window):-recent_window]
            else:
                baseline = similarities[:-recent_window]

        if len(baseline) == 0:
            return 0.0

        # Drift is the difference between baseline and recent average
        # Lower similarity = higher drift
        baseline_mean = np.mean(baseline)
        recent_mean = np.mean(recent)

        # Convert similarity drop to drift metric
        # If recent similarity is lower than baseline, that's drift
        drift = max(0.0, baseline_mean - recent_mean)

        return float(drift)

    def _calculate_long_term_drift(
        self,
        similarities: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[float, DriftTrend]:
        """
        Calculate long-term drift trend.

        Uses linear regression to find trend direction.
        """
        if len(similarities) < 3:
            return 0.0, DriftTrend(
                direction="stable",
                slope=0.0,
                confidence=0.0,
                samples_analyzed=len(similarities),
            )

        # Normalize timestamps to days
        time_days = (timestamps - timestamps[0]) / 86400

        # Linear regression
        n = len(similarities)
        x_mean = np.mean(time_days)
        y_mean = np.mean(similarities)

        numerator = np.sum((time_days - x_mean) * (similarities - y_mean))
        denominator = np.sum((time_days - x_mean) ** 2)

        if denominator < 1e-10:
            slope = 0.0
        else:
            slope = numerator / denominator

        # Determine trend direction
        # Negative slope = decreasing similarity = drift
        if abs(slope) < 0.001:  # Per day
            direction = "stable"
        elif slope < 0:
            direction = "decreasing"  # Similarity decreasing = drift increasing
        else:
            direction = "increasing"  # Similarity increasing = drift decreasing

        # Calculate R-squared for confidence
        y_pred = slope * (time_days - x_mean) + y_mean
        ss_res = np.sum((similarities - y_pred) ** 2)
        ss_tot = np.sum((similarities - y_mean) ** 2)

        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 0.0

        # Long-term drift based on range
        long_term_drift = float(max(0.0, np.max(similarities) - np.min(similarities)))

        return long_term_drift, DriftTrend(
            direction=direction,
            slope=float(slope),
            confidence=float(max(0, r_squared)),
            samples_analyzed=n,
        )

    def _detect_anomaly(
        self,
        similarities: np.ndarray,
        z_threshold: float = 3.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect anomalous sudden changes.

        Uses z-score to identify outliers.
        """
        if len(similarities) < 5:
            return False, None

        mean = np.mean(similarities)
        std = np.std(similarities)

        if std < 1e-10:
            return False, None

        # Check for outliers
        z_scores = np.abs((similarities - mean) / std)
        outliers = z_scores > z_threshold

        if np.any(outliers):
            # Find the most recent outlier
            outlier_indices = np.where(outliers)[0]
            most_recent = outlier_indices[-1]

            if most_recent >= len(similarities) - 3:
                # Outlier in recent samples
                outlier_value = similarities[most_recent]
                z = z_scores[most_recent]
                return True, f"Sudden change detected: similarity={outlier_value:.3f} (z-score={z:.1f})"

        # Check for sudden drops (even if not outliers)
        if len(similarities) >= 3:
            recent_diff = similarities[-1] - similarities[-3]
            if recent_diff < -self.anomaly_threshold:
                return True, f"Rapid similarity drop: {-recent_diff:.3f} in last 3 samples"

        return False, None

    def _determine_severity(
        self,
        short_term_drift: float,
        long_term_drift: float,
        anomaly_detected: bool,
    ) -> DriftSeverity:
        """Determine overall drift severity."""
        if anomaly_detected:
            return DriftSeverity.ANOMALOUS

        if long_term_drift > self.long_term_threshold:
            return DriftSeverity.SEVERE

        if short_term_drift > self.short_term_threshold:
            if short_term_drift > self.short_term_threshold * 1.5:
                return DriftSeverity.MODERATE
            return DriftSeverity.MINOR

        if long_term_drift > self.short_term_threshold:
            return DriftSeverity.MINOR

        return DriftSeverity.NONE

    def _generate_message(
        self,
        severity: DriftSeverity,
        short_term_drift: float,
        long_term_drift: float,
        anomaly_detected: bool,
        requires_re_enrollment: bool,
        update_recommended: bool,
    ) -> str:
        """Generate human-readable drift message."""
        if severity == DriftSeverity.NONE:
            return "Voice biometric stable, no significant drift detected"

        parts = []

        if anomaly_detected:
            parts.append("Anomalous change detected")

        if short_term_drift > 0.01:
            parts.append(f"Short-term drift: {short_term_drift:.1%}")

        if long_term_drift > 0.01:
            parts.append(f"Long-term drift: {long_term_drift:.1%}")

        if requires_re_enrollment:
            parts.append("Re-enrollment required")
        elif update_recommended:
            parts.append("Template update recommended")

        return "; ".join(parts) if parts else "Drift analysis complete"


class AdaptiveTemplateUpdater:
    """
    Adaptively update biometric templates to account for drift.

    Uses weighted averaging to incorporate new samples while
    maintaining template stability.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        min_confidence: float = 0.7,
        min_samples_before_update: int = 5,
    ):
        """
        Initialize adaptive updater.

        Args:
            learning_rate: How much to weight new samples (0.1 = 10%)
            min_confidence: Minimum verification confidence to allow update
            min_samples_before_update: Minimum successful verifications before updating
        """
        self.learning_rate = learning_rate
        self.min_confidence = min_confidence
        self.min_samples_before_update = min_samples_before_update

    def should_update(
        self,
        similarity: float,
        successful_count: int,
        drift_report: DriftReport,
    ) -> bool:
        """
        Determine if template should be updated.

        Args:
            similarity: Current verification similarity
            successful_count: Number of successful verifications since last update
            drift_report: Recent drift analysis

        Returns:
            True if update is recommended
        """
        # Don't update if drift is anomalous
        if drift_report.anomaly_detected:
            logger.warning("Skipping update due to anomaly detection")
            return False

        # Require minimum successful verifications
        if successful_count < self.min_samples_before_update:
            return False

        # Require minimum confidence
        if similarity < self.min_confidence:
            return False

        # Update if recommended or if there's minor drift
        if drift_report.update_recommended:
            return True

        if drift_report.severity == DriftSeverity.MINOR:
            return True

        return False

    def compute_updated_embedding(
        self,
        template_embedding: np.ndarray,
        new_embedding: np.ndarray,
        similarity: float,
    ) -> np.ndarray:
        """
        Compute updated template embedding.

        Uses exponential moving average weighted by similarity.

        Args:
            template_embedding: Current template
            new_embedding: New verified embedding
            similarity: Similarity score (higher = more weight to new)

        Returns:
            Updated embedding (L2 normalized)
        """
        # Weight learning rate by similarity
        # Higher similarity = more trust in new sample
        effective_lr = self.learning_rate * similarity

        # Exponential moving average
        updated = (1 - effective_lr) * template_embedding + effective_lr * new_embedding

        # L2 normalize
        norm = np.linalg.norm(updated)
        if norm > 0:
            updated = updated / norm

        return updated


def analyze_drift(
    voice_id: str,
    storage,
) -> DriftReport:
    """
    Convenience function to analyze drift for a voice.

    Args:
        voice_id: Voice ID
        storage: BiometricStorage instance

    Returns:
        DriftReport
    """
    history = storage.get_embedding_history(voice_id, limit=100)
    monitor = DriftMonitor()
    return monitor.analyze(voice_id, history)
