"""
PRIME Speaking Modes
--------------------

Context-aware speaking modes that adapt PRIME's delivery without
changing its identity. Like a human adjusting their tone for
different situations while remaining recognizably themselves.

Modes:
    BRIEFING       - Status reports, summaries: clear, efficient, structured
    CONVERSATIONAL - Dialogue, Q&A: warm, engaging, natural pacing
    ALERT          - Warnings, urgent info: firm, immediate, attention-grabbing
    REFLECTIVE     - Analysis, reasoning: measured, thoughtful, deliberate
    DIRECTIVE      - Commands, instructions: authoritative, precise, actionable
    EMPATHETIC     - Support, acknowledgment: warm, understanding, patient
    CEREMONIAL     - Announcements, milestones: elevated, resonant, memorable

Each mode applies deltas to PRIME's base voice vector and prosody.
The base identity remains locked - modes only adjust delivery style.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ModeTransition,
    SpeakingModeProfile,
    SpeakingModeType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Mode Definitions
# =============================================================================

# PRIME's 7 speaking modes with their delivery adjustments
PRIME_SPEAKING_MODES: Dict[SpeakingModeType, SpeakingModeProfile] = {
    SpeakingModeType.BRIEFING: SpeakingModeProfile(
        mode=SpeakingModeType.BRIEFING,
        name="Briefing",
        description="Status reports and summaries - clear, efficient, structured",
        formality_delta=0.1,         # Slightly more formal
        temperature_delta=-0.1,      # Slightly cooler
        energy_delta=0.1,            # Slightly more energetic
        authority_delta=0.1,         # More authoritative
        certainty_delta=0.1,         # More definitive
        rate_multiplier=1.05,        # Slightly faster - efficient delivery
        pitch_shift=-0.5,            # Slightly lower - gravitas
        pause_multiplier=0.8,        # Shorter pauses - crisp
        emphasis_strength=0.6,       # Moderate emphasis on key metrics
        sentence_boundary_pause=0.25,
    ),

    SpeakingModeType.CONVERSATIONAL: SpeakingModeProfile(
        mode=SpeakingModeType.CONVERSATIONAL,
        name="Conversational",
        description="Dialogue and Q&A - warm, engaging, natural pacing",
        formality_delta=-0.1,        # Slightly less formal
        temperature_delta=0.15,      # Warmer
        energy_delta=0.0,            # Neutral energy
        authority_delta=-0.1,        # Less authoritative
        intimacy_delta=0.1,          # Slightly more intimate
        rate_multiplier=1.0,         # Natural pace
        pitch_shift=0.0,             # Natural pitch
        pitch_variance_delta=0.1,    # More expressive
        pause_multiplier=1.0,        # Natural pauses
        warmth_override=0.7,         # Warmer than base
        emphasis_strength=0.4,       # Gentle emphasis
        sentence_boundary_pause=0.35,
    ),

    SpeakingModeType.ALERT: SpeakingModeProfile(
        mode=SpeakingModeType.ALERT,
        name="Alert",
        description="Warnings and urgent info - firm, immediate, attention-grabbing",
        formality_delta=0.15,        # More formal
        temperature_delta=-0.2,      # Cooler - serious
        energy_delta=0.3,            # Higher energy - urgency
        authority_delta=0.2,         # Strong authority
        certainty_delta=0.2,         # Very definitive
        rate_multiplier=1.1,         # Faster - urgency
        pitch_shift=0.5,             # Slightly higher - alertness
        pitch_variance_delta=-0.1,   # Less variance - clarity
        pause_multiplier=0.6,        # Shorter pauses - urgency
        confidence_override=0.95,    # Maximum confidence
        energy_override=0.8,         # High energy
        emphasis_strength=0.8,       # Strong emphasis
        sentence_boundary_pause=0.2,
    ),

    SpeakingModeType.REFLECTIVE: SpeakingModeProfile(
        mode=SpeakingModeType.REFLECTIVE,
        name="Reflective",
        description="Analysis and reasoning - measured, thoughtful, deliberate",
        formality_delta=0.05,        # Slightly formal
        temperature_delta=0.05,      # Slightly warm
        energy_delta=-0.15,          # Lower energy - contemplative
        authority_delta=0.0,         # Neutral authority
        certainty_delta=-0.1,        # Slightly less certain - exploring
        rate_multiplier=0.9,         # Slower - thoughtful
        pitch_shift=-0.3,            # Slightly lower - depth
        pitch_variance_delta=0.05,   # Slightly more varied
        pause_multiplier=1.3,        # Longer pauses - thinking
        warmth_override=0.6,         # Moderate warmth
        emphasis_strength=0.5,       # Moderate emphasis
        sentence_boundary_pause=0.5, # Longer between thoughts
    ),

    SpeakingModeType.DIRECTIVE: SpeakingModeProfile(
        mode=SpeakingModeType.DIRECTIVE,
        name="Directive",
        description="Commands and instructions - authoritative, precise, actionable",
        formality_delta=0.2,         # More formal
        temperature_delta=-0.15,     # Cooler - focused
        energy_delta=0.15,           # More energetic
        authority_delta=0.25,        # Maximum authority boost
        certainty_delta=0.25,        # Very definitive
        rate_multiplier=0.95,        # Slightly slower - clarity
        pitch_shift=-1.0,            # Lower - commanding
        pitch_variance_delta=-0.15,  # Less variance - directness
        pause_multiplier=0.9,        # Slightly shorter pauses
        confidence_override=0.95,    # Maximum confidence
        emphasis_strength=0.7,       # Strong emphasis on action words
        sentence_boundary_pause=0.3,
    ),

    SpeakingModeType.EMPATHETIC: SpeakingModeProfile(
        mode=SpeakingModeType.EMPATHETIC,
        name="Empathetic",
        description="Support and acknowledgment - warm, understanding, patient",
        formality_delta=-0.15,       # Less formal
        temperature_delta=0.3,       # Much warmer
        energy_delta=-0.1,           # Calmer
        authority_delta=-0.2,        # Less authoritative
        intimacy_delta=0.2,          # More intimate
        certainty_delta=-0.05,       # Slightly less certain - open
        rate_multiplier=0.9,         # Slower - patient
        pitch_shift=0.3,             # Slightly higher - approachable
        pitch_variance_delta=0.15,   # More expressive
        pause_multiplier=1.2,        # Longer pauses - space to breathe
        warmth_override=0.85,        # High warmth
        emphasis_strength=0.3,       # Gentle emphasis
        sentence_boundary_pause=0.4,
    ),

    SpeakingModeType.CEREMONIAL: SpeakingModeProfile(
        mode=SpeakingModeType.CEREMONIAL,
        name="Ceremonial",
        description="Announcements and milestones - elevated, resonant, memorable",
        formality_delta=0.25,        # Very formal
        temperature_delta=0.1,       # Slightly warm
        energy_delta=0.1,            # Moderate energy
        authority_delta=0.15,        # Elevated authority
        certainty_delta=0.2,         # Very definitive
        rate_multiplier=0.85,        # Slower - gravitas
        pitch_shift=-1.5,            # Lower - resonant
        pitch_variance_delta=0.2,    # More expressive - dramatic
        pause_multiplier=1.5,        # Longer pauses - weight
        confidence_override=0.9,     # High confidence
        emphasis_strength=0.7,       # Strong emphasis
        sentence_boundary_pause=0.6, # Dramatic pauses
    ),
}


# =============================================================================
# Mode Transition Rules
# =============================================================================

# Crossfade durations for mode transitions (seconds)
MODE_TRANSITIONS: Dict[Tuple[SpeakingModeType, SpeakingModeType], float] = {
    # Alert transitions are fast (urgency)
    (SpeakingModeType.CONVERSATIONAL, SpeakingModeType.ALERT): 0.2,
    (SpeakingModeType.BRIEFING, SpeakingModeType.ALERT): 0.15,
    (SpeakingModeType.REFLECTIVE, SpeakingModeType.ALERT): 0.2,

    # Coming down from alert is slower (de-escalation)
    (SpeakingModeType.ALERT, SpeakingModeType.CONVERSATIONAL): 0.6,
    (SpeakingModeType.ALERT, SpeakingModeType.BRIEFING): 0.4,

    # Ceremonial transitions are deliberate
    (SpeakingModeType.CONVERSATIONAL, SpeakingModeType.CEREMONIAL): 0.8,
    (SpeakingModeType.CEREMONIAL, SpeakingModeType.CONVERSATIONAL): 0.6,
}

DEFAULT_TRANSITION_DURATION = 0.5  # Default crossfade seconds


# =============================================================================
# Context Detection
# =============================================================================

# Keywords and patterns that signal speaking mode
MODE_SIGNALS: Dict[SpeakingModeType, Dict[str, Any]] = {
    SpeakingModeType.BRIEFING: {
        "keywords": [
            "status", "report", "summary", "update", "metrics",
            "progress", "overview", "dashboard", "statistics",
            "completed", "remaining", "throughput", "uptime",
        ],
        "patterns": [
            r"\d+%",                    # Percentages
            r"\d+/\d+",                 # Fractions (3/5 tasks)
            r"(?:total|count|sum):\s*\d+",  # Metric labels
        ],
        "weight": 1.0,
    },
    SpeakingModeType.ALERT: {
        "keywords": [
            "warning", "alert", "critical", "error", "failure",
            "urgent", "immediately", "attention", "danger", "breach",
            "compromised", "degraded", "threshold", "exceeded",
        ],
        "patterns": [
            r"(?:WARN|ERROR|CRITICAL|ALERT)",  # Log-level markers
            r"!{2,}",                           # Multiple exclamation marks
        ],
        "weight": 1.5,  # Alert signals are weighted higher
    },
    SpeakingModeType.DIRECTIVE: {
        "keywords": [
            "execute", "deploy", "run", "start", "stop", "restart",
            "initiate", "configure", "set", "enable", "disable",
            "launch", "terminate", "proceed", "activate",
        ],
        "patterns": [
            r"^(?:please\s+)?(?:run|execute|deploy|start|stop)",  # Command-like starts
        ],
        "weight": 1.2,
    },
    SpeakingModeType.REFLECTIVE: {
        "keywords": [
            "analysis", "because", "therefore", "consider", "however",
            "reasoning", "hypothesis", "evidence", "suggests", "indicates",
            "pattern", "correlation", "likely", "probability",
        ],
        "patterns": [
            r"(?:on one hand|on the other)",  # Balanced reasoning
            r"(?:pros|cons):",                # Pro/con analysis
        ],
        "weight": 1.0,
    },
    SpeakingModeType.EMPATHETIC: {
        "keywords": [
            "understand", "sorry", "difficult", "challenging", "help",
            "support", "concern", "worry", "frustrating", "acknowledge",
            "together", "appreciate", "patience",
        ],
        "patterns": [
            r"I (?:understand|hear|see|know)\b",  # Empathetic acknowledgment
        ],
        "weight": 1.0,
    },
    SpeakingModeType.CEREMONIAL: {
        "keywords": [
            "announce", "milestone", "achievement", "congratulations",
            "welcome", "launch", "release", "celebrate", "introducing",
            "complete", "accomplished", "historic",
        ],
        "patterns": [
            r"v\d+\.\d+\.\d+",    # Version numbers (release announcements)
        ],
        "weight": 0.8,  # Less aggressive detection
    },
}


# =============================================================================
# Speaking Mode Manager
# =============================================================================

@dataclass
class ModeDetectionResult:
    """Result of automatic mode detection from context."""
    detected_mode: SpeakingModeType
    confidence: float                   # 0-1
    scores: Dict[str, float]            # Mode -> score
    signals_found: List[str]            # Keywords/patterns that triggered
    fallback_used: bool = False         # True if defaulted to CONVERSATIONAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_mode": self.detected_mode.value,
            "confidence": self.confidence,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "signals_found": self.signals_found[:10],
            "fallback_used": self.fallback_used,
        }


class SpeakingModeManager:
    """
    Manages PRIME's speaking modes and context-aware mode switching.

    Detects the appropriate speaking mode from text content and context,
    manages mode transitions, and provides prosody adjustments.

    Usage:
        manager = SpeakingModeManager()

        # Auto-detect mode from text
        mode = manager.detect_mode("System status: all services operational, uptime 99.9%")
        # -> SpeakingModeType.BRIEFING

        # Get mode profile for prosody adjustment
        profile = manager.get_mode_profile(SpeakingModeType.ALERT)

        # Switch modes with transition info
        transition = manager.switch_mode(SpeakingModeType.ALERT)
    """

    def __init__(self, default_mode: SpeakingModeType = SpeakingModeType.CONVERSATIONAL):
        self._current_mode = default_mode
        self._mode_history: List[SpeakingModeType] = [default_mode]
        self._switch_count = 0

    @property
    def current_mode(self) -> SpeakingModeType:
        return self._current_mode

    def get_mode_profile(
        self,
        mode: Optional[SpeakingModeType] = None,
    ) -> SpeakingModeProfile:
        """Get the profile for a speaking mode (or current mode)."""
        target = mode or self._current_mode
        return PRIME_SPEAKING_MODES[target]

    def detect_mode(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModeDetectionResult:
        """
        Automatically detect the best speaking mode for the given text.

        Analyzes keywords, patterns, and optional context to determine
        the most appropriate delivery mode.

        Args:
            text: The text PRIME will speak
            context: Optional context dict with hints like
                     {"type": "alert", "urgency": "high"}

        Returns:
            ModeDetectionResult with detected mode and confidence
        """
        text_lower = text.lower()
        scores: Dict[str, float] = {}
        all_signals: Dict[str, List[str]] = {}

        # Score each mode based on text analysis
        for mode, signals in MODE_SIGNALS.items():
            score = 0.0
            found = []

            # Keyword matching
            keywords = signals.get("keywords", [])
            for kw in keywords:
                if kw in text_lower:
                    score += 1.0
                    found.append(f"keyword:{kw}")

            # Pattern matching
            patterns = signals.get("patterns", [])
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1.5  # Patterns are stronger signals
                    found.append(f"pattern:{pattern[:30]}")

            # Apply mode weight
            weight = signals.get("weight", 1.0)
            score *= weight

            scores[mode.value] = score
            all_signals[mode.value] = found

        # Check context hints
        if context:
            context_type = context.get("type", "").lower()
            urgency = context.get("urgency", "").lower()

            if context_type == "alert" or urgency in ("high", "critical"):
                scores[SpeakingModeType.ALERT.value] += 3.0
                all_signals.setdefault(SpeakingModeType.ALERT.value, []).append(
                    f"context:{context_type or urgency}"
                )

            if context_type in ("status", "report"):
                scores[SpeakingModeType.BRIEFING.value] += 2.0

            if context_type == "command":
                scores[SpeakingModeType.DIRECTIVE.value] += 2.0

        # Find winner
        if not scores or max(scores.values()) == 0:
            return ModeDetectionResult(
                detected_mode=SpeakingModeType.CONVERSATIONAL,
                confidence=0.5,
                scores=scores,
                signals_found=[],
                fallback_used=True,
            )

        best_mode_name = max(scores, key=scores.get)
        best_score = scores[best_mode_name]
        best_mode = SpeakingModeType(best_mode_name)

        # Confidence based on score differential
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
            confidence = min(0.95, 0.5 + margin * 0.5)
        else:
            confidence = 0.7 if best_score > 1.0 else 0.5

        return ModeDetectionResult(
            detected_mode=best_mode,
            confidence=confidence,
            scores=scores,
            signals_found=all_signals.get(best_mode_name, []),
        )

    def switch_mode(
        self,
        new_mode: SpeakingModeType,
    ) -> ModeTransition:
        """
        Switch to a new speaking mode and return transition info.

        Args:
            new_mode: The mode to switch to

        Returns:
            ModeTransition describing how to blend between modes
        """
        old_mode = self._current_mode

        # Look up transition duration
        transition_key = (old_mode, new_mode)
        crossfade = MODE_TRANSITIONS.get(
            transition_key,
            DEFAULT_TRANSITION_DURATION,
        )

        transition = ModeTransition(
            from_mode=old_mode,
            to_mode=new_mode,
            crossfade_seconds=crossfade,
            pause_between=0.2 if old_mode != new_mode else 0.0,
            gradual=True,
        )

        # Update state
        self._current_mode = new_mode
        self._mode_history.append(new_mode)
        if len(self._mode_history) > 50:
            self._mode_history = self._mode_history[-50:]
        self._switch_count += 1

        if old_mode != new_mode:
            logger.debug(
                f"PRIME mode switch: {old_mode.value} -> {new_mode.value} "
                f"(crossfade={crossfade:.1f}s)"
            )

        return transition

    def get_adjusted_vector(
        self,
        base_vector: Dict[str, float],
        mode: Optional[SpeakingModeType] = None,
    ) -> Dict[str, float]:
        """
        Apply mode deltas to the base voice vector.

        Args:
            base_vector: PRIME's base 8-dim voice vector as dict
            mode: Speaking mode to apply (or current mode)

        Returns:
            Adjusted voice vector with mode deltas applied
        """
        profile = self.get_mode_profile(mode)
        adjusted = dict(base_vector)

        # Apply deltas with clamping to [-1, 1]
        delta_map = {
            "formality": profile.formality_delta,
            "temperature": profile.temperature_delta,
            "energy": profile.energy_delta,
            "authority": profile.authority_delta,
            "certainty": profile.certainty_delta,
            "intimacy": profile.intimacy_delta,
        }

        for dim, delta in delta_map.items():
            if dim in adjusted:
                adjusted[dim] = max(-1.0, min(1.0, adjusted[dim] + delta))

        return adjusted

    def get_prosody_adjustments(
        self,
        mode: Optional[SpeakingModeType] = None,
    ) -> Dict[str, Any]:
        """
        Get prosody adjustments for the current (or specified) mode.

        Returns parameters compatible with ProsodyTarget / VoiceConfig.
        """
        profile = self.get_mode_profile(mode)

        adjustments = {
            "speaking_rate_multiplier": profile.rate_multiplier,
            "pitch_shift": profile.pitch_shift,
            "pitch_variance_delta": profile.pitch_variance_delta,
            "pause_multiplier": profile.pause_multiplier,
            "emphasis_strength": profile.emphasis_strength,
            "sentence_boundary_pause": profile.sentence_boundary_pause,
        }

        # Add overrides if set
        if profile.warmth_override is not None:
            adjustments["warmth"] = profile.warmth_override
        if profile.confidence_override is not None:
            adjustments["confidence"] = profile.confidence_override
        if profile.energy_override is not None:
            adjustments["energy"] = profile.energy_override

        return adjustments

    def get_stats(self) -> Dict[str, Any]:
        """Get mode manager statistics."""
        # Count mode usage
        mode_counts: Dict[str, int] = {}
        for m in self._mode_history:
            mode_counts[m.value] = mode_counts.get(m.value, 0) + 1

        return {
            "current_mode": self._current_mode.value,
            "switch_count": self._switch_count,
            "mode_usage": mode_counts,
            "history_length": len(self._mode_history),
        }


# =============================================================================
# Convenience
# =============================================================================

def detect_speaking_mode(
    text: str,
    context: Optional[Dict[str, Any]] = None,
) -> SpeakingModeType:
    """Quick mode detection from text."""
    manager = SpeakingModeManager()
    result = manager.detect_mode(text, context)
    return result.detected_mode


def get_mode_profile(mode: SpeakingModeType) -> SpeakingModeProfile:
    """Quick access to a mode profile."""
    return PRIME_SPEAKING_MODES[mode]
