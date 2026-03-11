"""
VØX Voice Genome - Models
-------------------------

Data models for complete voice DNA mapping.

The Voice Genome represents the complete signature of a voice:
- Biometric: Physical/health markers
- Psychometric: Mental/emotional state
- Sociometric: Social influence and dynamics
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple


class HealthRisk(Enum):
    """Health risk level indicators."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"


class ConfidenceLevel(Enum):
    """Confidence in assessment."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class EmotionalState(Enum):
    """Detected emotional state."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    ANXIOUS = "anxious"
    EXCITED = "excited"
    CALM = "calm"
    STRESSED = "stressed"


class AuthenticityLevel(Enum):
    """Voice authenticity assessment."""
    GENUINE = "genuine"
    MOSTLY_GENUINE = "mostly_genuine"
    UNCERTAIN = "uncertain"
    POSSIBLY_PERFORMED = "possibly_performed"
    PERFORMED = "performed"


class DominanceLevel(Enum):
    """Social dominance signaling."""
    SUBMISSIVE = "submissive"
    NEUTRAL = "neutral"
    ASSERTIVE = "assertive"
    DOMINANT = "dominant"
    COMMANDING = "commanding"


# =============================================================================
# Low-Level Acoustic Features
# =============================================================================

@dataclass
class AcousticFeatures:
    """
    Low-level acoustic features extracted from voice.

    These are the raw measurements used by higher-level analyzers.
    """
    # Fundamental frequency (pitch)
    f0_mean: float = 0.0          # Average pitch (Hz)
    f0_std: float = 0.0           # Pitch variation
    f0_min: float = 0.0           # Lowest pitch
    f0_max: float = 0.0           # Highest pitch
    f0_range: float = 0.0         # Pitch range

    # Voice quality measures
    jitter_percent: float = 0.0   # Pitch perturbation (%)
    shimmer_percent: float = 0.0  # Amplitude perturbation (%)
    hnr_db: float = 0.0           # Harmonics-to-noise ratio (dB)
    nhr: float = 0.0              # Noise-to-harmonics ratio

    # Formants (vocal tract resonances)
    f1_mean: float = 0.0          # First formant (Hz)
    f2_mean: float = 0.0          # Second formant (Hz)
    f3_mean: float = 0.0          # Third formant (Hz)
    f4_mean: float = 0.0          # Fourth formant (Hz)
    formant_dispersion: float = 0.0  # Formant spacing

    # Temporal features
    speaking_rate: float = 0.0    # Syllables per second
    pause_ratio: float = 0.0      # Ratio of pauses to speech
    mean_pause_duration: float = 0.0  # Average pause length (s)

    # Energy/intensity
    intensity_mean: float = 0.0   # Average loudness (dB)
    intensity_std: float = 0.0    # Loudness variation
    intensity_range: float = 0.0  # Dynamic range

    # Spectral features
    spectral_centroid: float = 0.0     # Brightness
    spectral_spread: float = 0.0       # Spectral width
    spectral_slope: float = 0.0        # High vs low frequency balance
    spectral_flux: float = 0.0         # Spectral change rate
    mfcc_means: List[float] = field(default_factory=list)  # MFCCs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pitch": {
                "f0_mean": self.f0_mean,
                "f0_std": self.f0_std,
                "f0_range": self.f0_range,
            },
            "voice_quality": {
                "jitter_percent": self.jitter_percent,
                "shimmer_percent": self.shimmer_percent,
                "hnr_db": self.hnr_db,
            },
            "formants": {
                "f1": self.f1_mean,
                "f2": self.f2_mean,
                "f3": self.f3_mean,
                "f4": self.f4_mean,
            },
            "temporal": {
                "speaking_rate": self.speaking_rate,
                "pause_ratio": self.pause_ratio,
            },
            "intensity": {
                "mean_db": self.intensity_mean,
                "range_db": self.intensity_range,
            },
        }


# =============================================================================
# Biometric Markers (Physical/Health)
# =============================================================================

@dataclass
class BiometricMarkers:
    """
    Physical and health-related markers from voice.

    Based on research linking voice characteristics to health conditions.
    """
    # Demographics (estimated)
    estimated_age: float = 0.0
    age_confidence: float = 0.0
    biological_sex: str = "unknown"  # male/female/unknown
    sex_confidence: float = 0.0

    # Vocal health
    vocal_fatigue: float = 0.0       # 0-1, current fatigue level
    vocal_strain: float = 0.0        # 0-1, vocal cord strain
    hydration_estimate: float = 0.5  # 0-1, hydration level

    # Respiratory markers
    breath_support: float = 0.5      # 0-1, diaphragmatic support
    respiratory_rate: float = 0.0    # Breaths per minute (estimated)
    respiratory_health: float = 0.5  # 0-1, overall respiratory score

    # Neurological markers
    motor_control: float = 0.5       # 0-1, fine motor control
    cognitive_load: float = 0.0      # 0-1, mental effort detected
    speech_fluency: float = 0.5      # 0-1, fluency score

    # Health risk indicators
    health_risks: Dict[str, HealthRisk] = field(default_factory=dict)
    health_notes: List[str] = field(default_factory=list)

    # Overall health score
    overall_health_score: float = 0.5  # 0-1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "demographics": {
                "estimated_age": self.estimated_age,
                "age_confidence": self.age_confidence,
                "biological_sex": self.biological_sex,
            },
            "vocal_health": {
                "fatigue": self.vocal_fatigue,
                "strain": self.vocal_strain,
                "hydration": self.hydration_estimate,
            },
            "respiratory": {
                "breath_support": self.breath_support,
                "respiratory_rate": self.respiratory_rate,
                "health_score": self.respiratory_health,
            },
            "neurological": {
                "motor_control": self.motor_control,
                "cognitive_load": self.cognitive_load,
                "speech_fluency": self.speech_fluency,
            },
            "overall_health_score": self.overall_health_score,
            "health_risks": {k: v.value for k, v in self.health_risks.items()},
            "notes": self.health_notes,
        }


# =============================================================================
# Psychometric Markers (Mental/Emotional)
# =============================================================================

@dataclass
class PsychometricMarkers:
    """
    Psychological and emotional state markers from voice.

    Detects current mental state and personality indicators.
    """
    # Current emotional state
    primary_emotion: EmotionalState = EmotionalState.NEUTRAL
    emotion_confidence: float = 0.0
    emotion_intensity: float = 0.0     # 0-1, how strong
    emotional_valence: float = 0.0     # -1 to 1 (negative to positive)
    emotional_arousal: float = 0.5     # 0-1 (calm to excited)

    # Stress and anxiety
    stress_level: float = 0.0          # 0-1
    anxiety_markers: float = 0.0       # 0-1
    tension_score: float = 0.0         # 0-1, vocal tension

    # Authenticity and deception
    authenticity: AuthenticityLevel = AuthenticityLevel.GENUINE
    authenticity_score: float = 0.8    # 0-1
    cognitive_load_speech: float = 0.0 # Extra processing (possible deception marker)
    confidence_in_speech: float = 0.5  # 0-1, how confident they sound

    # Mental state indicators
    mental_clarity: float = 0.5        # 0-1
    focus_level: float = 0.5           # 0-1
    engagement_level: float = 0.5      # 0-1

    # Personality indicators (Big Five proxy)
    extraversion_signal: float = 0.5   # 0-1
    openness_signal: float = 0.5       # 0-1
    agreeableness_signal: float = 0.5  # 0-1
    conscientiousness_signal: float = 0.5  # 0-1
    neuroticism_signal: float = 0.5    # 0-1

    # Historical patterns (if available)
    emotional_stability: float = 0.5   # 0-1, variance over time
    baseline_deviation: float = 0.0    # How far from personal baseline

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "emotional_state": {
                "primary": self.primary_emotion.value,
                "confidence": self.emotion_confidence,
                "intensity": self.emotion_intensity,
                "valence": self.emotional_valence,
                "arousal": self.emotional_arousal,
            },
            "stress_anxiety": {
                "stress_level": self.stress_level,
                "anxiety_markers": self.anxiety_markers,
                "tension": self.tension_score,
            },
            "authenticity": {
                "level": self.authenticity.value,
                "score": self.authenticity_score,
                "confidence_in_speech": self.confidence_in_speech,
            },
            "mental_state": {
                "clarity": self.mental_clarity,
                "focus": self.focus_level,
                "engagement": self.engagement_level,
            },
            "personality_signals": {
                "extraversion": self.extraversion_signal,
                "openness": self.openness_signal,
                "agreeableness": self.agreeableness_signal,
                "conscientiousness": self.conscientiousness_signal,
                "neuroticism": self.neuroticism_signal,
            },
        }


# =============================================================================
# Sociometric Markers (Social/Influence)
# =============================================================================

@dataclass
class SociometricMarkers:
    """
    Social dynamics and influence markers from voice.

    Measures how the voice affects others and social positioning.
    """
    # Authority and dominance
    dominance: DominanceLevel = DominanceLevel.NEUTRAL
    dominance_score: float = 0.5       # 0-1
    authority_signal: float = 0.5      # 0-1, perceived authority
    leadership_potential: float = 0.5  # 0-1

    # Warmth and approachability
    warmth_score: float = 0.5          # 0-1
    approachability: float = 0.5       # 0-1
    friendliness_signal: float = 0.5   # 0-1

    # Trust and credibility
    trust_signal: float = 0.5          # 0-1, how trustworthy they sound
    credibility_score: float = 0.5     # 0-1
    sincerity_signal: float = 0.5      # 0-1

    # Persuasion and influence
    persuasion_potential: float = 0.5  # 0-1
    charisma_score: float = 0.5        # 0-1
    engagement_power: float = 0.5      # 0-1, ability to hold attention

    # Social status signals
    perceived_status: float = 0.5      # 0-1
    education_signal: float = 0.5      # 0-1 (from speech patterns)
    professionalism: float = 0.5       # 0-1

    # Compatibility (for voice matching)
    compatibility_profile: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "authority": {
                "dominance": self.dominance.value,
                "dominance_score": self.dominance_score,
                "authority_signal": self.authority_signal,
                "leadership_potential": self.leadership_potential,
            },
            "warmth": {
                "warmth_score": self.warmth_score,
                "approachability": self.approachability,
                "friendliness": self.friendliness_signal,
            },
            "trust": {
                "trust_signal": self.trust_signal,
                "credibility": self.credibility_score,
                "sincerity": self.sincerity_signal,
            },
            "influence": {
                "persuasion_potential": self.persuasion_potential,
                "charisma": self.charisma_score,
                "engagement_power": self.engagement_power,
            },
            "status": {
                "perceived_status": self.perceived_status,
                "professionalism": self.professionalism,
            },
        }


# =============================================================================
# Complete Voice Genome
# =============================================================================

@dataclass
class VoiceGenome:
    """
    Complete Voice Genome - the full DNA map of a voice.

    Combines biometric, psychometric, and sociometric analysis
    into a comprehensive voice profile.
    """
    # Identification
    genome_id: str = ""
    subject_id: str = ""             # Optional identifier
    created_at: datetime = field(default_factory=datetime.now)

    # Audio metadata
    audio_duration: float = 0.0      # Seconds of audio analyzed
    audio_quality: float = 0.5       # 0-1, recording quality
    sample_rate: int = 0

    # Raw features
    acoustic_features: Optional[AcousticFeatures] = None

    # Analysis results
    biometric: Optional[BiometricMarkers] = None
    psychometric: Optional[PsychometricMarkers] = None
    sociometric: Optional[SociometricMarkers] = None

    # Overall scores
    voice_health_score: float = 0.5      # 0-1
    voice_authenticity_score: float = 0.5 # 0-1
    voice_influence_score: float = 0.5   # 0-1

    # Unique voice signature
    voice_signature: List[float] = field(default_factory=list)  # Embedding vector

    # Insights and recommendations
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Confidence in overall analysis
    analysis_confidence: ConfidenceLevel = ConfidenceLevel.MODERATE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": {
                "genome_id": self.genome_id,
                "subject_id": self.subject_id,
                "created_at": self.created_at.isoformat(),
                "audio_duration": self.audio_duration,
                "audio_quality": self.audio_quality,
                "analysis_confidence": self.analysis_confidence.value,
            },
            "acoustic_features": self.acoustic_features.to_dict() if self.acoustic_features else None,
            "biometric": self.biometric.to_dict() if self.biometric else None,
            "psychometric": self.psychometric.to_dict() if self.psychometric else None,
            "sociometric": self.sociometric.to_dict() if self.sociometric else None,
            "overall_scores": {
                "health": self.voice_health_score,
                "authenticity": self.voice_authenticity_score,
                "influence": self.voice_influence_score,
            },
            "insights": self.key_insights,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Voice Genome Analysis",
            "",
            f"**Genome ID:** `{self.genome_id}`",
            f"**Analyzed:** {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Audio Duration:** {self.audio_duration:.1f}s",
            f"**Confidence:** {self.analysis_confidence.value}",
            "",
            "---",
            "",
            "## Overall Scores",
            "",
            f"| Dimension | Score |",
            f"|-----------|-------|",
            f"| Health | {self.voice_health_score:.0%} |",
            f"| Authenticity | {self.voice_authenticity_score:.0%} |",
            f"| Influence Potential | {self.voice_influence_score:.0%} |",
            "",
        ]

        # Biometric section
        if self.biometric:
            b = self.biometric
            lines.extend([
                "## Biometric Analysis (Physical)",
                "",
                f"**Estimated Age:** {b.estimated_age:.0f} years "
                f"({b.age_confidence:.0%} confidence)",
                "",
                "| Marker | Value |",
                "|--------|-------|",
                f"| Vocal Fatigue | {b.vocal_fatigue:.0%} |",
                f"| Hydration | {b.hydration_estimate:.0%} |",
                f"| Breath Support | {b.breath_support:.0%} |",
                f"| Motor Control | {b.motor_control:.0%} |",
                f"| Speech Fluency | {b.speech_fluency:.0%} |",
                "",
            ])

            if b.health_notes:
                lines.append("**Health Notes:**")
                for note in b.health_notes:
                    lines.append(f"- {note}")
                lines.append("")

        # Psychometric section
        if self.psychometric:
            p = self.psychometric
            lines.extend([
                "## Psychometric Analysis (Mental/Emotional)",
                "",
                f"**Primary Emotion:** {p.primary_emotion.value.title()} "
                f"({p.emotion_confidence:.0%} confidence)",
                f"**Emotional Valence:** {p.emotional_valence:+.2f} "
                f"({'positive' if p.emotional_valence > 0 else 'negative' if p.emotional_valence < 0 else 'neutral'})",
                "",
                "| Marker | Value |",
                "|--------|-------|",
                f"| Stress Level | {p.stress_level:.0%} |",
                f"| Anxiety Markers | {p.anxiety_markers:.0%} |",
                f"| Authenticity | {p.authenticity.value} ({p.authenticity_score:.0%}) |",
                f"| Confidence in Speech | {p.confidence_in_speech:.0%} |",
                f"| Mental Clarity | {p.mental_clarity:.0%} |",
                "",
                "**Personality Signals (Big Five Proxy):**",
                "",
                f"- Extraversion: {p.extraversion_signal:.0%}",
                f"- Openness: {p.openness_signal:.0%}",
                f"- Agreeableness: {p.agreeableness_signal:.0%}",
                f"- Conscientiousness: {p.conscientiousness_signal:.0%}",
                f"- Neuroticism: {p.neuroticism_signal:.0%}",
                "",
            ])

        # Sociometric section
        if self.sociometric:
            s = self.sociometric
            lines.extend([
                "## Sociometric Analysis (Social/Influence)",
                "",
                f"**Dominance Style:** {s.dominance.value.title()}",
                "",
                "| Marker | Value |",
                "|--------|-------|",
                f"| Authority Signal | {s.authority_signal:.0%} |",
                f"| Leadership Potential | {s.leadership_potential:.0%} |",
                f"| Warmth | {s.warmth_score:.0%} |",
                f"| Trust Signal | {s.trust_signal:.0%} |",
                f"| Persuasion Potential | {s.persuasion_potential:.0%} |",
                f"| Charisma | {s.charisma_score:.0%} |",
                "",
            ])

        # Insights
        if self.key_insights:
            lines.extend([
                "## Key Insights",
                "",
            ])
            for insight in self.key_insights:
                lines.append(f"- {insight}")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.extend([
                "## ⚠️ Warnings",
                "",
            ])
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class GenomeComparison:
    """
    Comparison between two voice genomes.

    Used for compatibility analysis, change detection, etc.
    """
    genome_a_id: str
    genome_b_id: str
    comparison_type: str = "compatibility"  # compatibility, change, similarity

    # Similarity scores
    overall_similarity: float = 0.0
    acoustic_similarity: float = 0.0
    psychometric_similarity: float = 0.0
    sociometric_similarity: float = 0.0

    # Compatibility (for voice matching)
    compatibility_score: float = 0.0
    compatibility_notes: List[str] = field(default_factory=list)

    # Changes (for tracking over time)
    significant_changes: List[str] = field(default_factory=list)
    change_direction: str = ""  # improving, declining, stable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "genome_a": self.genome_a_id,
            "genome_b": self.genome_b_id,
            "type": self.comparison_type,
            "similarity": {
                "overall": self.overall_similarity,
                "acoustic": self.acoustic_similarity,
                "psychometric": self.psychometric_similarity,
                "sociometric": self.sociometric_similarity,
            },
            "compatibility": {
                "score": self.compatibility_score,
                "notes": self.compatibility_notes,
            },
            "changes": self.significant_changes,
        }
