"""
VØX Voice Genome - Sociometric Analyzer
---------------------------------------

Analyzes voice for social dynamics and influence markers.

Detects:
- Authority and dominance signals
- Warmth and approachability
- Trust and credibility
- Persuasion potential
- Social status signals
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

from .models import (
    AcousticFeatures,
    SociometricMarkers,
    DominanceLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class SociometricConfig:
    """
    Configuration for sociometric analysis.

    Attributes:
        analyze_authority: Detect authority/dominance signals
        analyze_warmth: Detect warmth/approachability
        analyze_trust: Detect trust/credibility signals
        analyze_persuasion: Detect persuasion potential
        sensitivity: Analysis sensitivity (0-1)
    """
    analyze_authority: bool = True
    analyze_warmth: bool = True
    analyze_trust: bool = True
    analyze_persuasion: bool = True
    sensitivity: float = 0.5


class SociometricAnalyzer:
    """
    Analyzes voice for social dynamics markers.

    Uses acoustic features to infer how a voice is likely
    to be perceived in social contexts.
    """

    def __init__(
        self,
        config: Optional[SociometricConfig] = None,
    ):
        """
        Initialize sociometric analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or SociometricConfig()

    def analyze(
        self,
        features: AcousticFeatures,
    ) -> SociometricMarkers:
        """
        Analyze acoustic features for sociometric markers.

        Args:
            features: Extracted acoustic features

        Returns:
            Sociometric markers
        """
        markers = SociometricMarkers()

        # Authority and dominance
        if self.config.analyze_authority:
            dom, dom_score = self._analyze_dominance(features)
            markers.dominance = dom
            markers.dominance_score = dom_score
            markers.authority_signal = self._analyze_authority(features)
            markers.leadership_potential = self._analyze_leadership(features)

        # Warmth and approachability
        if self.config.analyze_warmth:
            markers.warmth_score = self._analyze_warmth(features)
            markers.approachability = self._analyze_approachability(features)
            markers.friendliness_signal = self._analyze_friendliness(features)

        # Trust and credibility
        if self.config.analyze_trust:
            markers.trust_signal = self._analyze_trust(features)
            markers.credibility_score = self._analyze_credibility(features)
            markers.sincerity_signal = self._analyze_sincerity(features)

        # Persuasion and influence
        if self.config.analyze_persuasion:
            markers.persuasion_potential = self._analyze_persuasion(features)
            markers.charisma_score = self._analyze_charisma(features)
            markers.engagement_power = self._analyze_engagement_power(features)

        # Social status signals
        markers.perceived_status = self._analyze_status(features)
        markers.education_signal = self._analyze_education_signal(features)
        markers.professionalism = self._analyze_professionalism(features)

        # Compatibility profile
        markers.compatibility_profile = self._build_compatibility_profile(features, markers)

        return markers

    def _analyze_dominance(
        self,
        features: AcousticFeatures,
    ) -> Tuple[DominanceLevel, float]:
        """
        Analyze dominance signals in voice.

        Dominant voices tend to be:
        - Lower pitched (especially for males)
        - Louder
        - Less variable pitch (confident)
        - Slower, deliberate rate
        """
        score = 0.5

        # Lower pitch signals dominance
        if features.f0_mean < 120:
            score += 0.2
        elif features.f0_mean > 200:
            score -= 0.15

        # Higher intensity signals dominance
        if features.intensity_mean > -12:
            score += 0.15
        elif features.intensity_mean < -20:
            score -= 0.1

        # Lower pitch variability signals confidence/dominance
        if features.f0_std < 25:
            score += 0.1
        elif features.f0_std > 50:
            score -= 0.1

        # Slower, deliberate rate
        if features.speaking_rate < 4:
            score += 0.05

        # Map to level
        if score > 0.75:
            level = DominanceLevel.COMMANDING
        elif score > 0.6:
            level = DominanceLevel.DOMINANT
        elif score > 0.45:
            level = DominanceLevel.ASSERTIVE
        elif score > 0.3:
            level = DominanceLevel.NEUTRAL
        else:
            level = DominanceLevel.SUBMISSIVE

        return level, max(0.1, min(0.95, score))

    def _analyze_authority(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze perceived authority (0-1)."""
        authority = 0.5

        # Lower pitch
        if features.f0_mean < 130:
            authority += 0.15
        elif features.f0_mean > 180:
            authority -= 0.1

        # Steady, confident voice
        if features.f0_std < 30:
            authority += 0.15
        elif features.f0_std > 50:
            authority -= 0.15

        # Good volume
        if features.intensity_mean > -15:
            authority += 0.1

        # Clear voice (high HNR)
        if features.hnr_db > 22:
            authority += 0.1

        return max(0.1, min(1, authority))

    def _analyze_leadership(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze leadership potential (0-1)."""
        leadership = 0.5

        # Authority base
        leadership += (self._analyze_authority(features) - 0.5) * 0.5

        # Clarity and fluency
        if features.speaking_rate > 3.5:
            leadership += 0.1

        # Low hesitation
        if features.pause_ratio < 0.25:
            leadership += 0.1

        # Dynamic range (engaging)
        if features.intensity_range > 15:
            leadership += 0.1

        return max(0.1, min(1, leadership))

    def _analyze_warmth(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze perceived warmth (0-1)."""
        warmth = 0.5

        # Slightly higher pitch
        if features.f0_mean > 140:
            warmth += 0.1

        # More pitch variation (expressiveness)
        if features.f0_std > 30:
            warmth += 0.15

        # Clear, resonant voice
        if features.hnr_db > 20:
            warmth += 0.1

        # Not too loud (aggressive) or too quiet (distant)
        if -18 < features.intensity_mean < -10:
            warmth += 0.1

        # Natural speaking rate
        if 3.5 < features.speaking_rate < 5:
            warmth += 0.05

        return max(0.1, min(1, warmth))

    def _analyze_approachability(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze approachability (0-1)."""
        approachability = 0.5

        # Warmth base
        approachability += (self._analyze_warmth(features) - 0.5) * 0.4

        # Not too dominant
        if features.f0_mean > 130:
            approachability += 0.1

        # Pitch variation (friendly)
        if features.f0_std > 25:
            approachability += 0.1

        # Moderate intensity
        if features.intensity_mean < -12:
            approachability += 0.1

        return max(0.1, min(1, approachability))

    def _analyze_friendliness(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze friendliness signal (0-1)."""
        friendliness = 0.5

        # Higher pitch variation
        if features.f0_std > 30:
            friendliness += 0.2

        # Higher overall pitch
        if features.f0_mean > 150:
            friendliness += 0.1

        # Dynamic intensity (animated)
        if features.intensity_range > 15:
            friendliness += 0.1

        # Clear voice
        if features.hnr_db > 20:
            friendliness += 0.1

        return max(0.1, min(1, friendliness))

    def _analyze_trust(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze trust signal (0-1)."""
        trust = 0.5

        # Consistent, steady voice
        if features.f0_std < 35:
            trust += 0.15

        # Low jitter (reliable)
        if features.jitter_percent < 1:
            trust += 0.1

        # Clear voice
        if features.hnr_db > 20:
            trust += 0.1

        # Moderate rate (not rushed)
        if 3 < features.speaking_rate < 5:
            trust += 0.1

        # Appropriate pauses (thoughtful)
        if 0.15 < features.pause_ratio < 0.35:
            trust += 0.05

        return max(0.1, min(1, trust))

    def _analyze_credibility(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze perceived credibility (0-1)."""
        credibility = 0.5

        # Trust base
        credibility += (self._analyze_trust(features) - 0.5) * 0.5

        # Authority component
        credibility += (self._analyze_authority(features) - 0.5) * 0.3

        # Fluent speech
        if features.speaking_rate > 3:
            credibility += 0.1

        # Low hesitation
        if features.pause_ratio < 0.3:
            credibility += 0.1

        return max(0.1, min(1, credibility))

    def _analyze_sincerity(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze sincerity signal (0-1)."""
        sincerity = 0.6

        # Natural pitch variation (not monotone/scripted)
        if 20 < features.f0_std < 50:
            sincerity += 0.15

        # Natural jitter (not too perfect)
        if 0.3 < features.jitter_percent < 1.5:
            sincerity += 0.1

        # Clear voice
        if features.hnr_db > 18:
            sincerity += 0.1

        # Too perfect or too imperfect reduces sincerity
        if features.jitter_percent < 0.2 or features.jitter_percent > 2.5:
            sincerity -= 0.15

        return max(0.1, min(1, sincerity))

    def _analyze_persuasion(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze persuasion potential (0-1)."""
        persuasion = 0.5

        # Credibility base
        persuasion += (self._analyze_credibility(features) - 0.5) * 0.3

        # Warmth base
        persuasion += (self._analyze_warmth(features) - 0.5) * 0.2

        # Dynamic, engaging voice
        if features.intensity_range > 15:
            persuasion += 0.15

        # Good pitch variation
        if 25 < features.f0_std < 50:
            persuasion += 0.1

        # Clear articulation
        if features.hnr_db > 20:
            persuasion += 0.1

        return max(0.1, min(1, persuasion))

    def _analyze_charisma(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze charisma score (0-1)."""
        charisma = 0.5

        # Dynamic voice
        if features.intensity_range > 18:
            charisma += 0.15

        # Expressive pitch
        if features.f0_std > 30:
            charisma += 0.15

        # Good pace
        if features.speaking_rate > 4:
            charisma += 0.1

        # Clear, resonant
        if features.hnr_db > 22:
            charisma += 0.1

        # Confident (low hesitation)
        if features.pause_ratio < 0.25:
            charisma += 0.1

        return max(0.1, min(1, charisma))

    def _analyze_engagement_power(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze ability to engage and hold attention (0-1)."""
        engagement = 0.5

        # Dynamic intensity
        if features.intensity_range > 15:
            engagement += 0.2

        # Pitch variation
        if features.f0_std > 25:
            engagement += 0.15

        # Not monotonous
        if features.speaking_rate > 3:
            engagement += 0.1

        # Clear voice
        if features.hnr_db > 18:
            engagement += 0.05

        return max(0.1, min(1, engagement))

    def _analyze_status(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze perceived social status (0-1)."""
        status = 0.5

        # Authority signals
        status += (self._analyze_authority(features) - 0.5) * 0.4

        # Clear, educated speech patterns
        if features.hnr_db > 22:
            status += 0.1

        # Moderate, controlled rate
        if 3.5 < features.speaking_rate < 5:
            status += 0.1

        # Lower pitch for status
        if features.f0_mean < 140:
            status += 0.1

        return max(0.1, min(1, status))

    def _analyze_education_signal(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze education level signals (0-1)."""
        education = 0.5

        # Clear articulation
        if features.hnr_db > 22:
            education += 0.15

        # Moderate, measured rate
        if 3.5 < features.speaking_rate < 4.5:
            education += 0.1

        # Controlled pitch variation
        if 20 < features.f0_std < 40:
            education += 0.1

        # Low hesitation (fluent)
        if features.pause_ratio < 0.25:
            education += 0.1

        return max(0.1, min(1, education))

    def _analyze_professionalism(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Analyze professionalism (0-1)."""
        prof = 0.5

        # Controlled, steady voice
        if features.f0_std < 40:
            prof += 0.15

        # Clear voice
        if features.hnr_db > 20:
            prof += 0.1

        # Appropriate pace
        if 3.5 < features.speaking_rate < 5:
            prof += 0.1

        # Low hesitation
        if features.pause_ratio < 0.3:
            prof += 0.1

        # Consistent intensity
        if features.intensity_std < 6:
            prof += 0.1

        return max(0.1, min(1, prof))

    def _build_compatibility_profile(
        self,
        features: AcousticFeatures,
        markers: SociometricMarkers,
    ) -> Dict[str, float]:
        """Build voice compatibility profile for matching."""
        return {
            "authority_level": markers.authority_signal,
            "warmth_level": markers.warmth_score,
            "energy_level": min(1, features.speaking_rate / 6),
            "expressiveness": min(1, features.f0_std / 50),
            "clarity": min(1, features.hnr_db / 30),
            "pitch_band": self._get_pitch_band(features.f0_mean),
        }

    def _get_pitch_band(self, f0: float) -> float:
        """Normalize pitch to 0-1 band."""
        # 80-280 Hz range
        return max(0, min(1, (f0 - 80) / 200))


def analyze_sociometrics(
    features: AcousticFeatures,
    config: Optional[SociometricConfig] = None,
) -> SociometricMarkers:
    """
    Analyze voice features for sociometric markers.

    Args:
        features: Acoustic features
        config: Optional configuration

    Returns:
        Sociometric markers
    """
    analyzer = SociometricAnalyzer(config)
    return analyzer.analyze(features)
