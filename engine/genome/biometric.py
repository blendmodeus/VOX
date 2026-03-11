"""
VØX Voice Genome - Biometric Analyzer
-------------------------------------

Analyzes voice for physical and health-related markers.

Based on research linking voice characteristics to:
- Age and biological sex
- Vocal health (fatigue, strain)
- Respiratory health
- Neurological indicators
- General health markers
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from .models import (
    AcousticFeatures,
    BiometricMarkers,
    HealthRisk,
)

logger = logging.getLogger(__name__)


@dataclass
class BiometricConfig:
    """
    Configuration for biometric analysis.

    Attributes:
        estimate_age: Attempt age estimation
        estimate_sex: Attempt biological sex estimation
        detect_health_risks: Screen for health risk markers
        sensitivity: Analysis sensitivity (0-1)
    """
    estimate_age: bool = True
    estimate_sex: bool = True
    detect_health_risks: bool = True
    sensitivity: float = 0.5


class BiometricAnalyzer:
    """
    Analyzes voice for biometric/health markers.

    Uses acoustic features to infer physical characteristics
    and health indicators from voice.
    """

    # Reference values (based on research literature)
    # Male F0: 85-180 Hz (mean ~120 Hz)
    # Female F0: 165-255 Hz (mean ~210 Hz)
    MALE_F0_RANGE = (85, 180)
    FEMALE_F0_RANGE = (165, 255)

    # Normal jitter/shimmer thresholds
    NORMAL_JITTER_MAX = 1.0  # %
    NORMAL_SHIMMER_MAX = 3.0  # %
    NORMAL_HNR_MIN = 20.0  # dB

    def __init__(
        self,
        config: Optional[BiometricConfig] = None,
    ):
        """
        Initialize biometric analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or BiometricConfig()

    def analyze(
        self,
        features: AcousticFeatures,
    ) -> BiometricMarkers:
        """
        Analyze acoustic features for biometric markers.

        Args:
            features: Extracted acoustic features

        Returns:
            Biometric markers
        """
        markers = BiometricMarkers()

        # Estimate demographics
        if self.config.estimate_sex:
            sex, sex_conf = self._estimate_sex(features)
            markers.biological_sex = sex
            markers.sex_confidence = sex_conf

        if self.config.estimate_age:
            age, age_conf = self._estimate_age(features, markers.biological_sex)
            markers.estimated_age = age
            markers.age_confidence = age_conf

        # Vocal health
        markers.vocal_fatigue = self._estimate_vocal_fatigue(features)
        markers.vocal_strain = self._estimate_vocal_strain(features)
        markers.hydration_estimate = self._estimate_hydration(features)

        # Respiratory
        markers.breath_support = self._estimate_breath_support(features)
        markers.respiratory_rate = self._estimate_respiratory_rate(features)
        markers.respiratory_health = self._estimate_respiratory_health(features)

        # Neurological
        markers.motor_control = self._estimate_motor_control(features)
        markers.cognitive_load = self._estimate_cognitive_load(features)
        markers.speech_fluency = self._estimate_speech_fluency(features)

        # Health risk screening
        if self.config.detect_health_risks:
            markers.health_risks = self._screen_health_risks(features, markers)
            markers.health_notes = self._generate_health_notes(features, markers)

        # Overall health score
        markers.overall_health_score = self._calculate_health_score(markers)

        return markers

    def _estimate_sex(
        self,
        features: AcousticFeatures,
    ) -> tuple:
        """Estimate biological sex from F0."""
        f0 = features.f0_mean

        if f0 <= 0:
            return "unknown", 0.0

        # Calculate probability based on F0
        male_center = 130
        female_center = 210
        spread = 40

        male_prob = self._gaussian_prob(f0, male_center, spread)
        female_prob = self._gaussian_prob(f0, female_center, spread)

        total = male_prob + female_prob
        if total > 0:
            male_prob /= total
            female_prob /= total
        else:
            return "unknown", 0.0

        if male_prob > female_prob:
            return "male", male_prob
        else:
            return "female", female_prob

    def _gaussian_prob(self, x: float, mean: float, std: float) -> float:
        """Calculate Gaussian probability."""
        import math
        return math.exp(-0.5 * ((x - mean) / std) ** 2)

    def _estimate_age(
        self,
        features: AcousticFeatures,
        sex: str,
    ) -> tuple:
        """Estimate age from voice features."""
        # Age estimation is based on:
        # - F0 stability (jitter increases with age)
        # - Shimmer (increases with age)
        # - Speaking rate (decreases with age)
        # - HNR (decreases with age)

        # Base age from F0 (rough heuristic)
        if sex == "male":
            # Male F0 tends to rise slightly with age after 50
            if features.f0_mean < 100:
                base_age = 50
            elif features.f0_mean < 120:
                base_age = 35
            else:
                base_age = 25
        else:
            # Female F0 tends to drop after menopause
            if features.f0_mean < 180:
                base_age = 55
            elif features.f0_mean < 200:
                base_age = 40
            else:
                base_age = 30

        # Adjust for voice quality
        jitter_factor = min(1.0, features.jitter_percent / 2.0)  # Higher jitter = older
        shimmer_factor = min(1.0, features.shimmer_percent / 5.0)
        hnr_factor = max(0, 1 - features.hnr_db / 30)  # Lower HNR = older

        age_adjustment = (jitter_factor + shimmer_factor + hnr_factor) / 3 * 20

        estimated_age = base_age + age_adjustment

        # Clamp to reasonable range
        estimated_age = max(18, min(85, estimated_age))

        # Confidence based on consistency of markers
        confidence = 0.5  # Base confidence
        if features.hnr_db > 10:
            confidence += 0.2
        if features.jitter_percent < 3:
            confidence += 0.1
        if features.f0_std > 0:
            confidence += 0.1

        return estimated_age, min(0.9, confidence)

    def _estimate_vocal_fatigue(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate vocal fatigue level (0-1)."""
        fatigue = 0.0

        # Increased jitter indicates fatigue
        if features.jitter_percent > self.NORMAL_JITTER_MAX:
            fatigue += min(0.4, (features.jitter_percent - self.NORMAL_JITTER_MAX) / 3)

        # Increased shimmer indicates fatigue
        if features.shimmer_percent > self.NORMAL_SHIMMER_MAX:
            fatigue += min(0.3, (features.shimmer_percent - self.NORMAL_SHIMMER_MAX) / 5)

        # Reduced HNR indicates fatigue
        if features.hnr_db < self.NORMAL_HNR_MIN:
            fatigue += min(0.3, (self.NORMAL_HNR_MIN - features.hnr_db) / 20)

        return min(1.0, fatigue)

    def _estimate_vocal_strain(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate vocal strain level (0-1)."""
        strain = 0.0

        # High pitch variability can indicate strain
        if features.f0_std > 50:
            strain += min(0.3, (features.f0_std - 50) / 50)

        # Reduced HNR indicates strain
        if features.hnr_db < 15:
            strain += min(0.4, (15 - features.hnr_db) / 15)

        # High intensity variation
        if features.intensity_std > 10:
            strain += min(0.3, (features.intensity_std - 10) / 10)

        return min(1.0, strain)

    def _estimate_hydration(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate hydration level (0-1, 1=well hydrated)."""
        # Dehydration increases jitter and reduces HNR
        hydration = 0.7  # Base

        # Jitter impact
        if features.jitter_percent > 1.5:
            hydration -= min(0.3, (features.jitter_percent - 1.5) / 3)

        # HNR impact
        if features.hnr_db < 18:
            hydration -= min(0.3, (18 - features.hnr_db) / 18)

        # Shimmer impact
        if features.shimmer_percent > 4:
            hydration -= min(0.2, (features.shimmer_percent - 4) / 6)

        return max(0.1, min(1.0, hydration))

    def _estimate_breath_support(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate diaphragmatic breath support (0-1)."""
        support = 0.5

        # Consistent intensity suggests good support
        if features.intensity_std < 5:
            support += 0.2
        elif features.intensity_std > 15:
            support -= 0.2

        # Low pause ratio suggests good breath control
        if features.pause_ratio < 0.2:
            support += 0.2
        elif features.pause_ratio > 0.5:
            support -= 0.2

        # Speaking rate consistency
        if features.speaking_rate > 3 and features.speaking_rate < 6:
            support += 0.1

        return max(0.1, min(1.0, support))

    def _estimate_respiratory_rate(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate respiratory rate from pauses."""
        # This is a rough estimate based on speaking patterns
        # Normal: 12-20 breaths/minute

        if features.mean_pause_duration > 0 and features.pause_ratio > 0:
            # Estimate breaths per minute from pause patterns
            pause_frequency = features.pause_ratio / features.mean_pause_duration
            estimated_rate = pause_frequency * 60 / 10  # Rough scaling

            return max(8, min(30, estimated_rate))

        return 15.0  # Default

    def _estimate_respiratory_health(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate overall respiratory health (0-1)."""
        health = 0.7

        # Breath support
        support = self._estimate_breath_support(features)
        health = (health + support) / 2

        # HNR (respiratory issues often reduce HNR)
        if features.hnr_db > 20:
            health += 0.1
        elif features.hnr_db < 10:
            health -= 0.2

        # Pause patterns
        if features.pause_ratio > 0.6:
            health -= 0.1

        return max(0.1, min(1.0, health))

    def _estimate_motor_control(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate fine motor control (0-1)."""
        control = 0.7

        # Jitter reflects motor control
        if features.jitter_percent < 0.5:
            control += 0.2
        elif features.jitter_percent > 2:
            control -= 0.3

        # Speaking rate consistency
        if features.speaking_rate > 2 and features.speaking_rate < 7:
            control += 0.1

        return max(0.1, min(1.0, control))

    def _estimate_cognitive_load(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate cognitive load during speech (0-1)."""
        load = 0.3

        # More pauses = higher load
        if features.pause_ratio > 0.4:
            load += 0.3
        elif features.pause_ratio > 0.3:
            load += 0.2

        # Slower rate = higher load
        if features.speaking_rate < 3:
            load += 0.2

        # Pitch variation can indicate cognitive effort
        if features.f0_std > 40:
            load += 0.1

        return min(1.0, load)

    def _estimate_speech_fluency(
        self,
        features: AcousticFeatures,
    ) -> float:
        """Estimate speech fluency (0-1)."""
        fluency = 0.6

        # Speaking rate
        if features.speaking_rate > 4:
            fluency += 0.2
        elif features.speaking_rate < 2:
            fluency -= 0.2

        # Pause ratio
        if features.pause_ratio < 0.25:
            fluency += 0.2
        elif features.pause_ratio > 0.5:
            fluency -= 0.3

        return max(0.1, min(1.0, fluency))

    def _screen_health_risks(
        self,
        features: AcousticFeatures,
        markers: BiometricMarkers,
    ) -> Dict[str, HealthRisk]:
        """Screen for potential health risk markers."""
        risks = {}

        # Vocal cord issues (high jitter/shimmer)
        if features.jitter_percent > 2.0 or features.shimmer_percent > 6.0:
            risks["vocal_cord_stress"] = HealthRisk.MODERATE
        elif features.jitter_percent > 1.5 or features.shimmer_percent > 4.0:
            risks["vocal_cord_stress"] = HealthRisk.LOW

        # Respiratory concerns
        if markers.respiratory_health < 0.4:
            risks["respiratory"] = HealthRisk.MODERATE
        elif markers.respiratory_health < 0.6:
            risks["respiratory"] = HealthRisk.LOW

        # Fatigue
        if markers.vocal_fatigue > 0.7:
            risks["fatigue"] = HealthRisk.ELEVATED
        elif markers.vocal_fatigue > 0.5:
            risks["fatigue"] = HealthRisk.MODERATE

        # Motor control (potential neurological)
        if markers.motor_control < 0.3:
            risks["motor_control"] = HealthRisk.MODERATE
        elif markers.motor_control < 0.5:
            risks["motor_control"] = HealthRisk.LOW

        # Dehydration
        if markers.hydration_estimate < 0.3:
            risks["dehydration"] = HealthRisk.MODERATE
        elif markers.hydration_estimate < 0.5:
            risks["dehydration"] = HealthRisk.LOW

        return risks

    def _generate_health_notes(
        self,
        features: AcousticFeatures,
        markers: BiometricMarkers,
    ) -> List[str]:
        """Generate health-related notes."""
        notes = []

        if markers.vocal_fatigue > 0.5:
            notes.append("Voice shows signs of fatigue - consider vocal rest")

        if markers.hydration_estimate < 0.5:
            notes.append("Voice suggests possible dehydration - increase water intake")

        if markers.vocal_strain > 0.5:
            notes.append("Signs of vocal strain detected - avoid shouting/whispering")

        if features.jitter_percent > 1.5:
            notes.append("Elevated pitch perturbation - may indicate vocal fold irregularity")

        if features.hnr_db < 15:
            notes.append("Reduced voice clarity - consider voice evaluation if persistent")

        if markers.breath_support < 0.4:
            notes.append("Breath support could be improved - consider breathing exercises")

        return notes

    def _calculate_health_score(
        self,
        markers: BiometricMarkers,
    ) -> float:
        """Calculate overall voice health score."""
        scores = [
            1 - markers.vocal_fatigue,
            1 - markers.vocal_strain,
            markers.hydration_estimate,
            markers.breath_support,
            markers.respiratory_health,
            markers.motor_control,
            markers.speech_fluency,
        ]

        return sum(scores) / len(scores)


def analyze_biometrics(
    features: AcousticFeatures,
    config: Optional[BiometricConfig] = None,
) -> BiometricMarkers:
    """
    Analyze voice features for biometric markers.

    Args:
        features: Acoustic features
        config: Optional configuration

    Returns:
        Biometric markers
    """
    analyzer = BiometricAnalyzer(config)
    return analyzer.analyze(features)
