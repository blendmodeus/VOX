"""
VØX Resonance - Models
----------------------

Data models for song resonance analysis.

Analyzes audio + lyrics to predict psychological effects on listeners.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple


class MusicalKey(Enum):
    """Musical key classification."""
    C_MAJOR = "C major"
    C_MINOR = "C minor"
    D_MAJOR = "D major"
    D_MINOR = "D minor"
    E_MAJOR = "E major"
    E_MINOR = "E minor"
    F_MAJOR = "F major"
    F_MINOR = "F minor"
    G_MAJOR = "G major"
    G_MINOR = "G minor"
    A_MAJOR = "A major"
    A_MINOR = "A minor"
    B_MAJOR = "B major"
    B_MINOR = "B minor"
    UNKNOWN = "unknown"


class EmotionalValence(Enum):
    """Emotional valence spectrum."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class ArousalLevel(Enum):
    """Arousal/energy level."""
    VERY_LOW = "very_low"      # Sedating, sleep-inducing
    LOW = "low"                # Calming, relaxing
    MODERATE = "moderate"      # Balanced, steady
    HIGH = "high"              # Energizing, activating
    VERY_HIGH = "very_high"    # Intense, overwhelming


class ResonanceCategory(Enum):
    """Categories of psychological resonance."""
    EMOTIONAL = "emotional"           # Mood/feeling induction
    PHYSIOLOGICAL = "physiological"   # Body responses
    COGNITIVE = "cognitive"           # Thought patterns
    BEHAVIORAL = "behavioral"         # Action priming
    SOCIAL = "social"                 # Connection/bonding
    SPIRITUAL = "spiritual"           # Transcendence/meaning
    MEMORY = "memory"                 # Nostalgia/recall


class RiskLevel(Enum):
    """Risk assessment for listener effects."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class AudioFeatures:
    """
    Extracted audio features for resonance analysis.

    Attributes:
        tempo_bpm: Beats per minute
        key: Musical key (major/minor)
        time_signature: Time signature (e.g., "4/4")
        duration_seconds: Total duration
        loudness_db: Average loudness in dB
        dynamic_range_db: Difference between loudest and quietest
        bass_energy: Normalized bass frequency energy (0-1)
        mid_energy: Normalized mid frequency energy (0-1)
        treble_energy: Normalized treble frequency energy (0-1)
        spectral_centroid: Brightness measure
        spectral_complexity: Harmonic complexity score
        dissonance_score: Average dissonance (0-1)
        rhythm_regularity: How regular the rhythm is (0-1)
        vocal_presence: Estimated vocal presence (0-1)
        instrumental_sections: List of instrumental-only timestamps
    """
    tempo_bpm: float = 120.0
    key: MusicalKey = MusicalKey.UNKNOWN
    is_major: bool = True
    time_signature: str = "4/4"
    duration_seconds: float = 0.0
    loudness_db: float = -14.0
    dynamic_range_db: float = 10.0
    bass_energy: float = 0.5
    mid_energy: float = 0.5
    treble_energy: float = 0.5
    spectral_centroid: float = 2000.0
    spectral_complexity: float = 0.5
    dissonance_score: float = 0.2
    rhythm_regularity: float = 0.8
    vocal_presence: float = 0.7
    instrumental_sections: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tempo_bpm": self.tempo_bpm,
            "key": self.key.value,
            "is_major": self.is_major,
            "time_signature": self.time_signature,
            "duration_seconds": self.duration_seconds,
            "loudness_db": self.loudness_db,
            "dynamic_range_db": self.dynamic_range_db,
            "bass_energy": self.bass_energy,
            "mid_energy": self.mid_energy,
            "treble_energy": self.treble_energy,
            "spectral_centroid": self.spectral_centroid,
            "spectral_complexity": self.spectral_complexity,
            "dissonance_score": self.dissonance_score,
            "rhythm_regularity": self.rhythm_regularity,
            "vocal_presence": self.vocal_presence,
        }

    @property
    def energy_profile(self) -> str:
        """Describe energy distribution."""
        if self.bass_energy > 0.6:
            return "bass-heavy"
        elif self.treble_energy > 0.6:
            return "bright"
        elif self.mid_energy > 0.6:
            return "vocal-focused"
        else:
            return "balanced"

    @property
    def arousal_tendency(self) -> ArousalLevel:
        """Estimate arousal level from tempo and energy."""
        score = (self.tempo_bpm / 180) * 0.4 + \
                (self.bass_energy + self.loudness_db / -6) * 0.3 + \
                (1 - self.rhythm_regularity) * 0.3

        if score < 0.2:
            return ArousalLevel.VERY_LOW
        elif score < 0.4:
            return ArousalLevel.LOW
        elif score < 0.6:
            return ArousalLevel.MODERATE
        elif score < 0.8:
            return ArousalLevel.HIGH
        else:
            return ArousalLevel.VERY_HIGH


@dataclass
class LyricFeatures:
    """
    Extracted lyric features for resonance analysis.

    Attributes:
        raw_text: Original lyrics text
        word_count: Total word count
        unique_words: Number of unique words
        vocabulary_richness: Unique/total ratio
        sentiment_score: Overall sentiment (-1 to 1)
        emotional_valence: Categorical valence
        themes: Detected themes with confidence
        first_person_ratio: Ratio of I/me/my usage
        second_person_ratio: Ratio of you/your usage
        imperative_count: Number of commands
        question_count: Number of questions
        repetition_score: How repetitive (0-1)
        most_repeated: Most repeated phrases
        imagery_score: Vividness of imagery (0-1)
        abstraction_score: Abstract vs concrete (0-1)
        temporal_focus: Past/present/future distribution
        negation_ratio: Ratio of negative words
    """
    raw_text: str = ""
    word_count: int = 0
    unique_words: int = 0
    vocabulary_richness: float = 0.5
    sentiment_score: float = 0.0
    emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL
    themes: Dict[str, float] = field(default_factory=dict)
    first_person_ratio: float = 0.0
    second_person_ratio: float = 0.0
    imperative_count: int = 0
    question_count: int = 0
    repetition_score: float = 0.0
    most_repeated: List[str] = field(default_factory=list)
    imagery_score: float = 0.5
    abstraction_score: float = 0.5
    temporal_focus: Dict[str, float] = field(default_factory=lambda: {
        "past": 0.33, "present": 0.34, "future": 0.33
    })
    negation_ratio: float = 0.0
    emotional_words: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "word_count": self.word_count,
            "unique_words": self.unique_words,
            "vocabulary_richness": self.vocabulary_richness,
            "sentiment_score": self.sentiment_score,
            "emotional_valence": self.emotional_valence.value,
            "themes": self.themes,
            "first_person_ratio": self.first_person_ratio,
            "second_person_ratio": self.second_person_ratio,
            "repetition_score": self.repetition_score,
            "imagery_score": self.imagery_score,
            "temporal_focus": self.temporal_focus,
        }

    @property
    def identity_engagement(self) -> str:
        """How the lyrics engage listener identity."""
        if self.first_person_ratio > 0.15:
            return "personal_identification"
        elif self.second_person_ratio > 0.1:
            return "direct_address"
        else:
            return "observational"


@dataclass
class PsychologicalEffect:
    """
    A predicted psychological effect on listeners.

    Attributes:
        category: Type of effect
        name: Specific effect name
        description: What this effect does
        intensity: How strong (0-1)
        confidence: Prediction confidence (0-1)
        duration: Expected duration (transient/short/medium/long/lasting)
        mechanism: How this effect is produced
        evidence: Audio/lyric features supporting this
    """
    category: ResonanceCategory
    name: str
    description: str
    intensity: float = 0.5
    confidence: float = 0.7
    duration: str = "medium"
    mechanism: str = ""
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "name": self.name,
            "description": self.description,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "duration": self.duration,
            "mechanism": self.mechanism,
            "evidence": self.evidence,
        }


@dataclass
class ResonanceWarning:
    """
    Warning about potential negative effects.

    Attributes:
        risk_level: Severity of risk
        category: What type of risk
        description: What the warning is about
        triggers: What in the song triggers this
        vulnerable_groups: Who might be most affected
        recommendation: What to be aware of
    """
    risk_level: RiskLevel
    category: str
    description: str
    triggers: List[str] = field(default_factory=list)
    vulnerable_groups: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "risk_level": self.risk_level.value,
            "category": self.category,
            "description": self.description,
            "triggers": self.triggers,
            "vulnerable_groups": self.vulnerable_groups,
            "recommendation": self.recommendation,
        }


@dataclass
class ResonanceProfile:
    """
    Complete resonance profile for a song.

    Combines audio and lyric analysis into predicted effects.
    """
    # Metadata
    song_title: str = ""
    artist: str = ""
    analyzed_at: datetime = field(default_factory=datetime.now)

    # Feature extractions
    audio_features: Optional[AudioFeatures] = None
    lyric_features: Optional[LyricFeatures] = None

    # Overall assessments
    overall_valence: EmotionalValence = EmotionalValence.NEUTRAL
    overall_arousal: ArousalLevel = ArousalLevel.MODERATE
    overall_intensity: float = 0.5

    # Predicted effects
    primary_effects: List[PsychologicalEffect] = field(default_factory=list)
    secondary_effects: List[PsychologicalEffect] = field(default_factory=list)
    cumulative_effects: List[PsychologicalEffect] = field(default_factory=list)

    # Warnings
    warnings: List[ResonanceWarning] = field(default_factory=list)

    # Summary
    listener_impact_summary: str = ""
    recommended_contexts: List[str] = field(default_factory=list)
    contraindicated_contexts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": {
                "song_title": self.song_title,
                "artist": self.artist,
                "analyzed_at": self.analyzed_at.isoformat(),
            },
            "audio_features": self.audio_features.to_dict() if self.audio_features else None,
            "lyric_features": self.lyric_features.to_dict() if self.lyric_features else None,
            "overall": {
                "valence": self.overall_valence.value,
                "arousal": self.overall_arousal.value,
                "intensity": self.overall_intensity,
            },
            "effects": {
                "primary": [e.to_dict() for e in self.primary_effects],
                "secondary": [e.to_dict() for e in self.secondary_effects],
                "cumulative": [e.to_dict() for e in self.cumulative_effects],
            },
            "warnings": [w.to_dict() for w in self.warnings],
            "summary": {
                "impact": self.listener_impact_summary,
                "recommended_for": self.recommended_contexts,
                "not_recommended_for": self.contraindicated_contexts,
            },
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Resonance Analysis: {self.song_title}",
            f"**Artist:** {self.artist}",
            f"**Analyzed:** {self.analyzed_at.strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
            "## Overall Profile",
            "",
            f"- **Emotional Valence:** {self.overall_valence.value.replace('_', ' ').title()}",
            f"- **Arousal Level:** {self.overall_arousal.value.replace('_', ' ').title()}",
            f"- **Intensity:** {self.overall_intensity:.0%}",
            "",
        ]

        # Audio features summary
        if self.audio_features:
            af = self.audio_features
            lines.extend([
                "## Audio Characteristics",
                "",
                f"- **Tempo:** {af.tempo_bpm:.0f} BPM",
                f"- **Key:** {af.key.value}",
                f"- **Energy Profile:** {af.energy_profile}",
                f"- **Dynamic Range:** {af.dynamic_range_db:.1f} dB",
                f"- **Harmonic Complexity:** {af.spectral_complexity:.0%}",
                f"- **Dissonance:** {af.dissonance_score:.0%}",
                "",
            ])

        # Lyric features summary
        if self.lyric_features:
            lf = self.lyric_features
            lines.extend([
                "## Lyric Characteristics",
                "",
                f"- **Sentiment:** {lf.sentiment_score:+.2f}",
                f"- **Identity Engagement:** {lf.identity_engagement}",
                f"- **Repetition:** {lf.repetition_score:.0%}",
                f"- **Imagery:** {lf.imagery_score:.0%}",
            ])
            if lf.themes:
                top_themes = sorted(lf.themes.items(), key=lambda x: -x[1])[:3]
                lines.append(f"- **Top Themes:** {', '.join(t[0] for t in top_themes)}")
            lines.append("")

        # Primary effects
        if self.primary_effects:
            lines.extend([
                "## What This Song Will Do To Listeners",
                "",
                "### Primary Effects",
                "",
            ])
            for effect in self.primary_effects:
                lines.append(f"**{effect.name}** ({effect.intensity:.0%} intensity)")
                lines.append(f"> {effect.description}")
                lines.append("")

        # Secondary effects
        if self.secondary_effects:
            lines.extend([
                "### Secondary Effects",
                "",
            ])
            for effect in self.secondary_effects:
                lines.append(f"- **{effect.name}:** {effect.description}")
            lines.append("")

        # Cumulative effects
        if self.cumulative_effects:
            lines.extend([
                "### With Repeated Listening",
                "",
            ])
            for effect in self.cumulative_effects:
                lines.append(f"- **{effect.name}:** {effect.description}")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.extend([
                "## ⚠️ Warnings",
                "",
            ])
            for warning in self.warnings:
                level_emoji = {
                    RiskLevel.LOW: "🟡",
                    RiskLevel.MODERATE: "🟠",
                    RiskLevel.HIGH: "🔴",
                    RiskLevel.SEVERE: "⛔",
                }.get(warning.risk_level, "⚪")
                lines.append(f"{level_emoji} **{warning.category}** ({warning.risk_level.value})")
                lines.append(f"> {warning.description}")
                if warning.recommendation:
                    lines.append(f"> *Recommendation: {warning.recommendation}*")
                lines.append("")

        # Summary
        if self.listener_impact_summary:
            lines.extend([
                "## Summary",
                "",
                self.listener_impact_summary,
                "",
            ])

        # Contexts
        if self.recommended_contexts:
            lines.extend([
                "### Recommended For",
                "",
            ])
            for ctx in self.recommended_contexts:
                lines.append(f"- {ctx}")
            lines.append("")

        if self.contraindicated_contexts:
            lines.extend([
                "### Not Recommended For",
                "",
            ])
            for ctx in self.contraindicated_contexts:
                lines.append(f"- {ctx}")
            lines.append("")

        return "\n".join(lines)
