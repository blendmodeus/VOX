"""
VØX Voice Genome - Synthesizer
------------------------------

Main entry point for complete Voice Genome analysis.

Combines:
- Feature extraction
- Biometric analysis
- Psychometric analysis
- Sociometric analysis

Into a complete Voice Genome profile.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np

from .models import (
    VoiceGenome,
    AcousticFeatures,
    BiometricMarkers,
    PsychometricMarkers,
    SociometricMarkers,
    ConfidenceLevel,
    GenomeComparison,
)
from .extractor import VoiceFeatureExtractor, ExtractorConfig
from .biometric import BiometricAnalyzer, BiometricConfig
from .psychometric import PsychometricAnalyzer, PsychometricConfig
from .sociometric import SociometricAnalyzer, SociometricConfig

logger = logging.getLogger(__name__)


@dataclass
class GenomeSynthesizerConfig:
    """
    Configuration for genome synthesizer.

    Attributes:
        extractor_config: Feature extraction config
        biometric_config: Biometric analysis config
        psychometric_config: Psychometric analysis config
        sociometric_config: Sociometric analysis config
        generate_insights: Auto-generate insights
        generate_recommendations: Auto-generate recommendations
        min_audio_duration: Minimum audio duration for reliable analysis
    """
    extractor_config: Optional[ExtractorConfig] = None
    biometric_config: Optional[BiometricConfig] = None
    psychometric_config: Optional[PsychometricConfig] = None
    sociometric_config: Optional[SociometricConfig] = None
    generate_insights: bool = True
    generate_recommendations: bool = True
    min_audio_duration: float = 3.0  # seconds


class VoiceGenomeSynthesizer:
    """
    Main synthesizer for complete Voice Genome analysis.

    Orchestrates feature extraction and all analysis modules
    to produce a comprehensive voice profile.
    """

    def __init__(
        self,
        config: Optional[GenomeSynthesizerConfig] = None,
    ):
        """
        Initialize genome synthesizer.

        Args:
            config: Synthesizer configuration
        """
        self.config = config or GenomeSynthesizerConfig()

        # Initialize components
        self.extractor = VoiceFeatureExtractor(self.config.extractor_config)
        self.biometric_analyzer = BiometricAnalyzer(self.config.biometric_config)
        self.psychometric_analyzer = PsychometricAnalyzer(self.config.psychometric_config)
        self.sociometric_analyzer = SociometricAnalyzer(self.config.sociometric_config)

    def analyze(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: Optional[int] = None,
        subject_id: str = "",
    ) -> VoiceGenome:
        """
        Perform complete Voice Genome analysis.

        Args:
            audio: Audio data or path to audio file
            sample_rate: Sample rate (required if audio is array)
            subject_id: Optional subject identifier

        Returns:
            Complete Voice Genome
        """
        genome = VoiceGenome(
            genome_id=self._generate_genome_id(),
            subject_id=subject_id,
            created_at=datetime.now(),
        )

        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio_path = str(audio)
            try:
                if hasattr(self.extractor, '_load_audio'):
                    audio_data, sample_rate = self.extractor._load_audio(audio_path)
                else:
                    audio_data = np.zeros(22050 * 10)
                    sample_rate = 22050
            except Exception as e:
                logger.warning(f"Failed to load audio: {e}")
                audio_data = np.zeros(22050 * 10)
                sample_rate = 22050
        else:
            audio_data = audio
            sample_rate = sample_rate or 22050

        # Calculate audio metadata
        genome.audio_duration = len(audio_data) / sample_rate
        genome.sample_rate = sample_rate
        genome.audio_quality = self._estimate_audio_quality(audio_data, sample_rate)

        # Check minimum duration
        if genome.audio_duration < self.config.min_audio_duration:
            logger.warning(
                f"Audio duration ({genome.audio_duration:.1f}s) below minimum "
                f"({self.config.min_audio_duration}s). Results may be unreliable."
            )
            genome.analysis_confidence = ConfidenceLevel.LOW
        else:
            genome.analysis_confidence = ConfidenceLevel.MODERATE

        # Extract acoustic features
        try:
            genome.acoustic_features = self.extractor.extract(audio_data, sample_rate)
            logger.info("Acoustic features extracted successfully")
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            genome.acoustic_features = AcousticFeatures()

        # Biometric analysis
        try:
            genome.biometric = self.biometric_analyzer.analyze(genome.acoustic_features)
            logger.info("Biometric analysis complete")
        except Exception as e:
            logger.warning(f"Biometric analysis failed: {e}")
            genome.biometric = BiometricMarkers()

        # Psychometric analysis
        try:
            genome.psychometric = self.psychometric_analyzer.analyze(genome.acoustic_features)
            logger.info("Psychometric analysis complete")
        except Exception as e:
            logger.warning(f"Psychometric analysis failed: {e}")
            genome.psychometric = PsychometricMarkers()

        # Sociometric analysis
        try:
            genome.sociometric = self.sociometric_analyzer.analyze(genome.acoustic_features)
            logger.info("Sociometric analysis complete")
        except Exception as e:
            logger.warning(f"Sociometric analysis failed: {e}")
            genome.sociometric = SociometricMarkers()

        # Calculate overall scores
        genome.voice_health_score = self._calculate_health_score(genome)
        genome.voice_authenticity_score = self._calculate_authenticity_score(genome)
        genome.voice_influence_score = self._calculate_influence_score(genome)

        # Generate voice signature (embedding)
        genome.voice_signature = self._generate_signature(genome)

        # Generate insights and recommendations
        if self.config.generate_insights:
            genome.key_insights = self._generate_insights(genome)

        if self.config.generate_recommendations:
            genome.recommendations = self._generate_recommendations(genome)
            genome.warnings = self._generate_warnings(genome)

        # Update confidence based on quality
        if genome.audio_quality > 0.7 and genome.audio_duration > 10:
            genome.analysis_confidence = ConfidenceLevel.HIGH

        return genome

    def analyze_file(
        self,
        path: str,
        subject_id: str = "",
    ) -> VoiceGenome:
        """Analyze audio file."""
        return self.analyze(path, subject_id=subject_id)

    def compare(
        self,
        genome_a: VoiceGenome,
        genome_b: VoiceGenome,
        comparison_type: str = "compatibility",
    ) -> GenomeComparison:
        """
        Compare two voice genomes.

        Args:
            genome_a: First genome
            genome_b: Second genome
            comparison_type: Type of comparison (compatibility, change, similarity)

        Returns:
            Comparison result
        """
        comparison = GenomeComparison(
            genome_a_id=genome_a.genome_id,
            genome_b_id=genome_b.genome_id,
            comparison_type=comparison_type,
        )

        # Calculate acoustic similarity
        if genome_a.acoustic_features and genome_b.acoustic_features:
            comparison.acoustic_similarity = self._acoustic_similarity(
                genome_a.acoustic_features,
                genome_b.acoustic_features,
            )

        # Calculate psychometric similarity
        if genome_a.psychometric and genome_b.psychometric:
            comparison.psychometric_similarity = self._psychometric_similarity(
                genome_a.psychometric,
                genome_b.psychometric,
            )

        # Calculate sociometric similarity
        if genome_a.sociometric and genome_b.sociometric:
            comparison.sociometric_similarity = self._sociometric_similarity(
                genome_a.sociometric,
                genome_b.sociometric,
            )

        # Overall similarity
        comparison.overall_similarity = (
            comparison.acoustic_similarity * 0.4 +
            comparison.psychometric_similarity * 0.3 +
            comparison.sociometric_similarity * 0.3
        )

        # Compatibility analysis
        if comparison_type == "compatibility":
            comparison.compatibility_score = self._calculate_compatibility(
                genome_a, genome_b
            )
            comparison.compatibility_notes = self._generate_compatibility_notes(
                genome_a, genome_b
            )

        # Change analysis
        elif comparison_type == "change":
            comparison.significant_changes = self._detect_changes(genome_a, genome_b)
            comparison.change_direction = self._determine_change_direction(
                genome_a, genome_b
            )

        return comparison

    def _generate_genome_id(self) -> str:
        """Generate unique genome ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    def _estimate_audio_quality(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float:
        """Estimate audio recording quality."""
        quality = 0.5

        # Check for clipping
        max_amp = np.max(np.abs(audio))
        if max_amp > 0.99:
            quality -= 0.2
        elif max_amp < 0.1:
            quality -= 0.1

        # Check sample rate
        if sample_rate >= 44100:
            quality += 0.2
        elif sample_rate >= 22050:
            quality += 0.1
        elif sample_rate < 16000:
            quality -= 0.1

        # Check for DC offset
        dc_offset = np.abs(np.mean(audio))
        if dc_offset > 0.1:
            quality -= 0.1

        return max(0.1, min(1.0, quality))

    def _calculate_health_score(self, genome: VoiceGenome) -> float:
        """Calculate overall voice health score."""
        if genome.biometric:
            return genome.biometric.overall_health_score
        return 0.5

    def _calculate_authenticity_score(self, genome: VoiceGenome) -> float:
        """Calculate overall authenticity score."""
        if genome.psychometric:
            return genome.psychometric.authenticity_score
        return 0.5

    def _calculate_influence_score(self, genome: VoiceGenome) -> float:
        """Calculate overall influence potential score."""
        if genome.sociometric:
            # Combine persuasion, charisma, and credibility
            s = genome.sociometric
            return (s.persuasion_potential + s.charisma_score + s.credibility_score) / 3
        return 0.5

    def _generate_signature(self, genome: VoiceGenome) -> List[float]:
        """Generate voice signature embedding."""
        signature = []

        # Acoustic features
        if genome.acoustic_features:
            af = genome.acoustic_features
            signature.extend([
                af.f0_mean / 300,  # Normalized pitch
                af.f0_std / 50,    # Normalized pitch variation
                af.jitter_percent / 5,
                af.shimmer_percent / 10,
                af.hnr_db / 30,
                af.speaking_rate / 7,
                af.intensity_mean / -10,  # Normalize
            ])

        # Psychometric features
        if genome.psychometric:
            pm = genome.psychometric
            signature.extend([
                pm.stress_level,
                pm.authenticity_score,
                pm.emotional_valence,
                pm.emotional_arousal,
            ])

        # Sociometric features
        if genome.sociometric:
            sm = genome.sociometric
            signature.extend([
                sm.dominance_score,
                sm.warmth_score,
                sm.trust_signal,
                sm.charisma_score,
            ])

        # Pad or truncate to fixed size
        target_size = 32
        if len(signature) < target_size:
            signature.extend([0.0] * (target_size - len(signature)))
        else:
            signature = signature[:target_size]

        return signature

    def _generate_insights(self, genome: VoiceGenome) -> List[str]:
        """Generate key insights from analysis."""
        insights = []

        # Biometric insights
        if genome.biometric:
            b = genome.biometric
            if b.vocal_fatigue > 0.6:
                insights.append("Voice shows significant fatigue - vocal rest recommended")
            if b.hydration_estimate < 0.4:
                insights.append("Voice suggests dehydration - increase water intake")
            if b.motor_control < 0.5:
                insights.append("Reduced motor control detected in speech patterns")

        # Psychometric insights
        if genome.psychometric:
            p = genome.psychometric
            if p.stress_level > 0.6:
                insights.append(f"Elevated stress detected ({p.stress_level:.0%})")
            if p.authenticity_score < 0.5:
                insights.append("Voice patterns suggest possible performance/scripting")
            if p.primary_emotion.value != "neutral":
                insights.append(
                    f"Primary emotional state: {p.primary_emotion.value} "
                    f"({p.emotion_confidence:.0%} confidence)"
                )

        # Sociometric insights
        if genome.sociometric:
            s = genome.sociometric
            if s.charisma_score > 0.7:
                insights.append("High charisma potential detected in voice")
            if s.authority_signal > 0.7:
                insights.append("Strong authority signals in voice pattern")
            if s.warmth_score > 0.7:
                insights.append("Voice projects high warmth and approachability")

        # Overall insights
        if genome.voice_influence_score > 0.7:
            insights.append("Voice has high influence potential")
        elif genome.voice_influence_score < 0.3:
            insights.append("Voice influence markers are below average")

        return insights[:7]  # Limit to top 7

    def _generate_recommendations(self, genome: VoiceGenome) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if genome.biometric:
            b = genome.biometric
            if b.vocal_fatigue > 0.5:
                recommendations.append("Consider vocal rest periods between extended speaking")
            if b.breath_support < 0.5:
                recommendations.append("Breathing exercises could improve voice projection")
            if b.hydration_estimate < 0.5:
                recommendations.append("Increase hydration for better vocal quality")

        if genome.psychometric:
            p = genome.psychometric
            if p.stress_level > 0.5:
                recommendations.append("Relaxation techniques may improve voice quality")
            if p.confidence_in_speech < 0.5:
                recommendations.append("Practice speaking to build vocal confidence")

        if genome.sociometric:
            s = genome.sociometric
            if s.warmth_score < 0.4:
                recommendations.append("Varying pitch more can increase perceived warmth")
            if s.authority_signal < 0.4 and s.dominance.value == "submissive":
                recommendations.append(
                    "Speaking slightly slower and lower can increase authority"
                )

        return recommendations[:5]

    def _generate_warnings(self, genome: VoiceGenome) -> List[str]:
        """Generate warnings based on analysis."""
        warnings = []

        if genome.biometric:
            b = genome.biometric
            for risk, level in b.health_risks.items():
                if level.value in ("elevated", "high"):
                    warnings.append(f"Health risk indicator: {risk} ({level.value})")

        if genome.audio_quality < 0.4:
            warnings.append("Low audio quality may affect analysis accuracy")

        if genome.audio_duration < 5:
            warnings.append("Short audio sample - results may be less reliable")

        return warnings

    def _acoustic_similarity(
        self,
        a: AcousticFeatures,
        b: AcousticFeatures,
    ) -> float:
        """Calculate acoustic similarity between two feature sets."""
        diffs = [
            abs(a.f0_mean - b.f0_mean) / 100,
            abs(a.f0_std - b.f0_std) / 30,
            abs(a.speaking_rate - b.speaking_rate) / 3,
            abs(a.intensity_mean - b.intensity_mean) / 20,
            abs(a.hnr_db - b.hnr_db) / 20,
        ]
        avg_diff = sum(diffs) / len(diffs)
        return max(0, 1 - avg_diff)

    def _psychometric_similarity(
        self,
        a: PsychometricMarkers,
        b: PsychometricMarkers,
    ) -> float:
        """Calculate psychometric similarity."""
        diffs = [
            abs(a.stress_level - b.stress_level),
            abs(a.emotional_valence - b.emotional_valence) / 2,
            abs(a.emotional_arousal - b.emotional_arousal),
            abs(a.authenticity_score - b.authenticity_score),
        ]
        avg_diff = sum(diffs) / len(diffs)
        return max(0, 1 - avg_diff)

    def _sociometric_similarity(
        self,
        a: SociometricMarkers,
        b: SociometricMarkers,
    ) -> float:
        """Calculate sociometric similarity."""
        diffs = [
            abs(a.dominance_score - b.dominance_score),
            abs(a.warmth_score - b.warmth_score),
            abs(a.trust_signal - b.trust_signal),
            abs(a.charisma_score - b.charisma_score),
        ]
        avg_diff = sum(diffs) / len(diffs)
        return max(0, 1 - avg_diff)

    def _calculate_compatibility(
        self,
        a: VoiceGenome,
        b: VoiceGenome,
    ) -> float:
        """Calculate voice compatibility score."""
        # Compatibility often benefits from complementary traits
        compatibility = 0.5

        if a.sociometric and b.sociometric:
            # Complementary dominance (one higher, one lower)
            dom_diff = abs(a.sociometric.dominance_score - b.sociometric.dominance_score)
            if 0.2 < dom_diff < 0.5:
                compatibility += 0.15

            # Similar warmth levels work well
            warmth_diff = abs(a.sociometric.warmth_score - b.sociometric.warmth_score)
            if warmth_diff < 0.2:
                compatibility += 0.15

            # Trust alignment
            trust_diff = abs(a.sociometric.trust_signal - b.sociometric.trust_signal)
            if trust_diff < 0.2:
                compatibility += 0.1

        if a.psychometric and b.psychometric:
            # Similar emotional baseline
            arousal_diff = abs(a.psychometric.emotional_arousal - b.psychometric.emotional_arousal)
            if arousal_diff < 0.25:
                compatibility += 0.1

        return min(1, compatibility)

    def _generate_compatibility_notes(
        self,
        a: VoiceGenome,
        b: VoiceGenome,
    ) -> List[str]:
        """Generate compatibility notes."""
        notes = []

        if a.sociometric and b.sociometric:
            if a.sociometric.dominance_score > 0.6 and b.sociometric.dominance_score > 0.6:
                notes.append("Both voices project dominance - potential for conflict")
            if a.sociometric.warmth_score > 0.6 and b.sociometric.warmth_score > 0.6:
                notes.append("Both voices are warm - good for collaboration")

        return notes

    def _detect_changes(
        self,
        old: VoiceGenome,
        new: VoiceGenome,
    ) -> List[str]:
        """Detect significant changes between genomes."""
        changes = []

        if old.biometric and new.biometric:
            fatigue_change = new.biometric.vocal_fatigue - old.biometric.vocal_fatigue
            if abs(fatigue_change) > 0.2:
                direction = "increased" if fatigue_change > 0 else "decreased"
                changes.append(f"Vocal fatigue {direction}")

        if old.psychometric and new.psychometric:
            stress_change = new.psychometric.stress_level - old.psychometric.stress_level
            if abs(stress_change) > 0.2:
                direction = "increased" if stress_change > 0 else "decreased"
                changes.append(f"Stress level {direction}")

        return changes

    def _determine_change_direction(
        self,
        old: VoiceGenome,
        new: VoiceGenome,
    ) -> str:
        """Determine overall change direction."""
        if new.voice_health_score > old.voice_health_score + 0.1:
            return "improving"
        elif new.voice_health_score < old.voice_health_score - 0.1:
            return "declining"
        return "stable"


def analyze_voice_genome(
    audio_path: str,
    subject_id: str = "",
    config: Optional[GenomeSynthesizerConfig] = None,
) -> VoiceGenome:
    """
    Analyze audio file to create Voice Genome.

    Args:
        audio_path: Path to audio file
        subject_id: Optional subject identifier
        config: Optional configuration

    Returns:
        Complete Voice Genome
    """
    synthesizer = VoiceGenomeSynthesizer(config)
    return synthesizer.analyze_file(audio_path, subject_id)


def generate_genome_report(
    audio_path: str,
    subject_id: str = "",
) -> str:
    """
    Generate Voice Genome report as markdown.

    Args:
        audio_path: Path to audio file
        subject_id: Optional subject identifier

    Returns:
        Markdown report
    """
    genome = analyze_voice_genome(audio_path, subject_id)
    return genome.to_markdown()
