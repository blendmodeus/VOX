"""
VØX Voice Genome - Psychometric Analyzer
----------------------------------------

Analyzes voice for psychological and emotional markers.

Detects:
- Current emotional state
- Stress and anxiety levels
- Authenticity/deception indicators
- Mental state markers
- Personality signals
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

from .models import (
    AcousticFeatures,
    PsychometricMarkers,
    EmotionalState,
    AuthenticityLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class PsychometricConfig:
    """
    Configuration for psychometric analysis.

    Attributes:
        detect_emotions: Enable emotion detection
        detect_stress: Enable stress/anxiety detection
        assess_authenticity: Enable authenticity assessment
        personality_inference: Enable personality signal detection
        sensitivity: Analysis sensitivity (0-1)
    """
    detect_emotions: bool = True
    detect_stress: bool = True
    assess_authenticity: bool = True
    personality_inference: bool = True
    sensitivity: float = 0.5


class PsychometricAnalyzer:
    """
    Analyzes voice for psychological markers.

    Uses acoustic features to infer emotional state,
    stress levels, authenticity, and personality traits.
    """

    # Reference values for emotion detection
    # Based on Scherer's component process model and related research

    def __init__(
        self,
        config: Optional[PsychometricConfig] = None,
    ):
        """
        Initialize psychometric analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or PsychometricConfig()

    def analyze(
        self,
        features: AcousticFeatures,
    ) -> PsychometricMarkers:
        """
        Analyze acoustic features for psychometric markers.

        Args:
            features: Extracted acoustic features

        Returns:
            Psychometric markers
        """
        markers = PsychometricMarkers()

        # Emotional state detection
        if self.config.detect_emotions:
            emotion, confidence, intensity = self._detect_emotion(features)
            markers.primary_emotion = emotion
            markers.emotion_confidence = confidence
            markers.emotion_intensity = intensity
            markers.emotional_valence = self._calculate_valence(emotion, features)
            markers.emotional_arousal = self._calculate_arousal(features)

        # Stress and anxiety
        if self.config.detect_stress:
            markers.stress_level = self._estimate_stress(features)
            markers.anxiety_markers = self._estimate_anxiety(features)
            markers.tension_score = self._estimate_tension(features)

        # Authenticity assessment
        if self.config.assess_authenticity:
            auth, score = self._assess_authenticity(features)
            markers.authenticity = auth
            markers.authenticity_score = score
            markers.cognitive_load_speech = self._estimate_speech_cognitive_load(features)
            markers.confidence_in_speech = self._estimate_speech_confidence(features)

        # Mental state
        markers.mental_clarity = self._estimate_mental_clarity(features)
        markers.focus_level = self._estimate_focus(features)
        markers.engagement_level = self._estimate_engagement(features)

        # Personality signals
        if self.config.personality_inference:
            personality = self._infer_personality_signals(features)
            markers.extraversion_signal = personality["extraversion"]
            markers.openness_signal = personality["openness"]
            markers.agreeableness_signal = personality["agreeableness"]
            markers.conscientiousness_signal = personality["conscientiousness"]
            markers.neuroticism_signal = personality["neuroticism"]

        # Stability metrics
        markers.emotional_stability = self._estimate_emotional_stability(features)

        return markers

    def _detect_emotion(
        self,
        features: AcousticFeatures,
    ) -> Tuple[EmotionalState, float, float]:
        """
        Detect primary emotional state.

        Based on research linking acoustic features to emotions:
        - Happy: High F0, high F0 variability, fast rate, high intensity
        - Sad: Low F0, low variability, slow rate, low intensity
        - Angry: High F0, high intensity, fast rate
        - Fear: High F0, high variability, fast rate
        - Neutral: Moderate values across all features

        Returns:
            Tuple of (emotion, confidence, intensity)
        """
        # Calculate emotion scores based on acoustic patterns
        scores = {}

        f0 = features.f0_mean
        f0_var = features.f0_std
        rate = features.speaking_rate
        intensity = features.intensity_mean

        # Normalize features to 0-1 scale
        f0_norm = min(1, max(0, (f0 - 100) / 200))  # 100-300 Hz range
        f0_var_norm = min(1, max(0, f0_var / 50))
        rate_norm = min(1, max(0, (rate - 2) / 5))  # 2-7 syll/s
        intensity_norm = min(1, max(0, (intensity + 30) / 30))  # -30 to 0 dB

        # Calculate emotion probabilities
        # Happy: high pitch, variable, fast, loud
        scores[EmotionalState.HAPPY] = (
            f0_norm * 0.3 +
            f0_var_norm * 0.3 +
            rate_norm * 0.2 +
            intensity_norm * 0.2
        )

        # Sad: low pitch, monotone, slow, quiet
        scores[EmotionalState.SAD] = (
            (1 - f0_norm) * 0.3 +
            (1 - f0_var_norm) * 0.3 +
            (1 - rate_norm) * 0.2 +
            (1 - intensity_norm) * 0.2
        )

        # Angry: high pitch, loud, fast
        scores[EmotionalState.ANGRY] = (
            f0_norm * 0.3 +
            intensity_norm * 0.4 +
            rate_norm * 0.3
        )

        # Fearful: high pitch, variable, fast
        scores[EmotionalState.FEARFUL] = (
            f0_norm * 0.3 +
            f0_var_norm * 0.4 +
            rate_norm * 0.3
        )

        # Anxious: variable pitch, faster rate
        scores[EmotionalState.ANXIOUS] = (
            f0_var_norm * 0.5 +
            rate_norm * 0.3 +
            (1 - intensity_norm) * 0.2
        )

        # Calm: low variability, moderate rate
        scores[EmotionalState.CALM] = (
            (1 - f0_var_norm) * 0.4 +
            (1 - abs(rate_norm - 0.5) * 2) * 0.3 +
            0.3  # Base calm probability
        )

        # Stressed: high variability, jitter/shimmer
        jitter_factor = min(1, features.jitter_percent / 2)
        scores[EmotionalState.STRESSED] = (
            f0_var_norm * 0.3 +
            jitter_factor * 0.4 +
            rate_norm * 0.3
        )

        # Neutral: moderate everything
        scores[EmotionalState.NEUTRAL] = (
            (1 - abs(f0_norm - 0.5) * 2) * 0.25 +
            (1 - abs(rate_norm - 0.5) * 2) * 0.25 +
            (1 - abs(intensity_norm - 0.5) * 2) * 0.25 +
            (1 - f0_var_norm) * 0.25
        )

        # Find primary emotion
        primary = max(scores, key=scores.get)
        confidence = scores[primary]

        # Intensity is based on deviation from neutral
        intensity = 1 - scores[EmotionalState.NEUTRAL]

        return primary, min(0.95, confidence), min(1.0, intensity)

    def _calculate_valence(
        self,
        emotion: EmotionalState,
        features: AcousticFeatures,
    ) -> float:
        """Calculate emotional valence (-1 to 1)."""
        # Emotion-based valence
        valence_map = {
            EmotionalState.HAPPY: 0.8,
            EmotionalState.EXCITED: 0.7,
            EmotionalState.CALM: 0.3,
            EmotionalState.NEUTRAL: 0.0,
            EmotionalState.ANXIOUS: -0.4,
            EmotionalState.SAD: -0.7,
            EmotionalState.ANGRY: -0.6,
            EmotionalState.FEARFUL: -0.8,
            EmotionalState.STRESSED: -0.5,
            EmotionalState.SURPRISED: 0.2,
            EmotionalState.DISGUSTED: -0.7,
        }

        base_valence = valence_map.get(emotion, 0.0)

        # Adjust based on acoustic features
        # Higher F0 and variability often indicate more positive valence
        f0_adjustment = (features.f0_mean - 150) / 200 * 0.2
        var_adjustment = features.f0_std / 100 * 0.1

        return max(-1, min(1, base_valence + f0_adjustment + var_adjustment))

    def _calculate_arousal(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Calculate emotional arousal (0-1)."""
        arousal = 0.5

        # Speaking rate affects arousal
        if features.speaking_rate > 5:
            arousal += 0.2
        elif features.speaking_rate < 3:
            arousal -= 0.2

        # F0 variability affects arousal
        if features.f0_std > 40:
            arousal += 0.2

        # Intensity affects arousal
        if features.intensity_mean > -10:
            arousal += 0.1

        return max(0, min(1, arousal))

    def _estimate_stress(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate stress level (0-1)."""
        stress = 0.2  # Base

        # Jitter increases with stress
        if features.jitter_percent > 1:
            stress += min(0.3, features.jitter_percent / 3)

        # F0 variability increases with stress
        if features.f0_std > 30:
            stress += min(0.2, (features.f0_std - 30) / 50)

        # Speaking rate changes under stress
        if features.speaking_rate > 5.5 or features.speaking_rate < 2.5:
            stress += 0.2

        # Reduced HNR under stress
        if features.hnr_db < 18:
            stress += min(0.2, (18 - features.hnr_db) / 18)

        return min(1, stress)

    def _estimate_anxiety(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate anxiety markers (0-1)."""
        anxiety = 0.1

        # High pitch variability
        if features.f0_std > 35:
            anxiety += min(0.3, (features.f0_std - 35) / 30)

        # Fast speaking rate
        if features.speaking_rate > 5:
            anxiety += min(0.2, (features.speaking_rate - 5) / 3)

        # More pauses (hesitation)
        if features.pause_ratio > 0.4:
            anxiety += 0.2

        # Shimmer (voice tremor)
        if features.shimmer_percent > 4:
            anxiety += min(0.2, (features.shimmer_percent - 4) / 6)

        return min(1, anxiety)

    def _estimate_tension(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate vocal tension (0-1)."""
        tension = 0.2

        # High pitch
        if features.f0_mean > 180:
            tension += min(0.3, (features.f0_mean - 180) / 100)

        # Jitter/shimmer
        tension += min(0.2, features.jitter_percent / 4)
        tension += min(0.2, features.shimmer_percent / 8)

        # Reduced HNR
        if features.hnr_db < 20:
            tension += min(0.2, (20 - features.hnr_db) / 20)

        return min(1, tension)

    def _assess_authenticity(
        self,
        features: AcousticFeatures,
    ) -> Tuple[AuthenticityLevel, float]:
        """
        Assess voice authenticity.

        Genuine speech typically has:
        - Natural pitch variation
        - Consistent but not monotone intensity
        - Appropriate pauses
        - Low cognitive load markers
        """
        score = 0.7  # Start with assuming genuine

        # Too monotone might indicate rehearsed/performed
        if features.f0_std < 15:
            score -= 0.15

        # Too variable might indicate uncertainty/performance
        if features.f0_std > 60:
            score -= 0.1

        # Very regular pauses might indicate reading
        if features.pause_ratio > 0.05 and features.mean_pause_duration > 0:
            pause_regularity = 1 / (1 + features.pause_ratio * features.mean_pause_duration)
            if pause_regularity > 0.8:
                score -= 0.1

        # Natural speech has some jitter (but not too much)
        if features.jitter_percent < 0.2:  # Too perfect
            score -= 0.1
        elif features.jitter_percent > 2.5:  # Too imperfect (stress/deception)
            score -= 0.15

        # Speaking rate consistency
        # (would need per-utterance analysis for better assessment)

        # Map score to level
        if score > 0.7:
            level = AuthenticityLevel.GENUINE
        elif score > 0.55:
            level = AuthenticityLevel.MOSTLY_GENUINE
        elif score > 0.4:
            level = AuthenticityLevel.UNCERTAIN
        elif score > 0.25:
            level = AuthenticityLevel.POSSIBLY_PERFORMED
        else:
            level = AuthenticityLevel.PERFORMED

        return level, max(0.1, min(0.95, score))

    def _estimate_speech_cognitive_load(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate cognitive load during speech."""
        load = 0.3

        # More pauses = more thinking
        if features.pause_ratio > 0.35:
            load += 0.3
        elif features.pause_ratio > 0.25:
            load += 0.15

        # Slower rate = more processing
        if features.speaking_rate < 3.5:
            load += 0.2

        # Higher pitch (stress of mental effort)
        if features.f0_mean > 170:
            load += 0.1

        return min(1, load)

    def _estimate_speech_confidence(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate how confident the speaker sounds."""
        confidence = 0.5

        # Lower pitch variance = more confident
        if features.f0_std < 25:
            confidence += 0.2
        elif features.f0_std > 50:
            confidence -= 0.2

        # Fewer pauses = more confident
        if features.pause_ratio < 0.2:
            confidence += 0.2
        elif features.pause_ratio > 0.4:
            confidence -= 0.2

        # Higher intensity = more confident
        if features.intensity_mean > -12:
            confidence += 0.1

        return max(0.1, min(1, confidence))

    def _estimate_mental_clarity(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate mental clarity (0-1)."""
        clarity = 0.6

        # Good speech fluency
        if features.speaking_rate > 3.5 and features.speaking_rate < 5.5:
            clarity += 0.2

        # Low pause ratio
        if features.pause_ratio < 0.25:
            clarity += 0.1

        # Low jitter (steady voice)
        if features.jitter_percent < 1:
            clarity += 0.1

        return min(1, clarity)

    def _estimate_focus(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate focus level (0-1)."""
        focus = 0.5

        # Consistent intensity
        if features.intensity_std < 5:
            focus += 0.2

        # Moderate speaking rate
        if 3.5 < features.speaking_rate < 5.5:
            focus += 0.15

        # Low pause ratio
        if features.pause_ratio < 0.3:
            focus += 0.15

        return min(1, focus)

    def _estimate_engagement(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate engagement level (0-1)."""
        engagement = 0.5

        # Pitch variation indicates engagement
        if features.f0_std > 25:
            engagement += 0.2

        # Dynamic intensity
        if features.intensity_range > 15:
            engagement += 0.15

        # Not too slow
        if features.speaking_rate > 3:
            engagement += 0.15

        return min(1, engagement)

    def _infer_personality_signals(
        self,
        features: AcousticFeatures,
    ) -> Dict[str, float]:
        """
        Infer Big Five personality signals from voice.

        Note: These are signals/tendencies, not definitive measurements.
        """
        # Extraversion: louder, faster, more variable
        extraversion = 0.5
        if features.intensity_mean > -12:
            extraversion += 0.15
        if features.speaking_rate > 4.5:
            extraversion += 0.15
        if features.f0_std > 35:
            extraversion += 0.1

        # Openness: more expressive, variable
        openness = 0.5
        if features.f0_std > 30:
            openness += 0.2
        if features.intensity_range > 20:
            openness += 0.15

        # Agreeableness: warmer, softer
        agreeableness = 0.5
        if features.f0_mean > 140:  # Slightly higher pitch
            agreeableness += 0.1
        if features.hnr_db > 22:  # Clearer voice
            agreeableness += 0.15

        # Conscientiousness: more measured, consistent
        conscientiousness = 0.5
        if features.f0_std < 35:
            conscientiousness += 0.15
        if features.intensity_std < 5:
            conscientiousness += 0.15
        if 3.5 < features.speaking_rate < 5:
            conscientiousness += 0.1

        # Neuroticism: more variable, tense
        neuroticism = 0.5
        if features.jitter_percent > 1:
            neuroticism += 0.15
        if features.f0_std > 45:
            neuroticism += 0.15
        if features.hnr_db < 18:
            neuroticism += 0.1

        return {
            "extraversion": max(0.1, min(0.9, extraversion)),
            "openness": max(0.1, min(0.9, openness)),
            "agreeableness": max(0.1, min(0.9, agreeableness)),
            "conscientiousness": max(0.1, min(0.9, conscientiousness)),
            "neuroticism": max(0.1, min(0.9, neuroticism)),
        }

    def _estimate_emotional_stability(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate emotional stability (0-1)."""
        stability = 0.6

        # Low pitch variance = more stable
        if features.f0_std < 30:
            stability += 0.2
        elif features.f0_std > 50:
            stability -= 0.2

        # Low jitter = more stable
        if features.jitter_percent < 0.8:
            stability += 0.1
        elif features.jitter_percent > 2:
            stability -= 0.1

        return max(0.1, min(1, stability))


def analyze_psychometrics(
    features: AcousticFeatures,
    config: Optional[PsychometricConfig] = None,
) -> PsychometricMarkers:
    """
    Analyze voice features for psychometric markers.

    Args:
        features: Acoustic features
        config: Optional configuration

    Returns:
        Psychometric markers
    """
    analyzer = PsychometricAnalyzer(config)
    return analyzer.analyze(features)
