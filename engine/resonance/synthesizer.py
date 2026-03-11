"""
VØX Resonance - Synthesizer
---------------------------

Synthesizes audio and lyric analysis into complete resonance profiles.

This is the main entry point for song analysis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np

from .models import (
    AudioFeatures,
    LyricFeatures,
    ResonanceProfile,
    PsychologicalEffect,
    ResonanceWarning,
    ResonanceCategory,
    EmotionalValence,
    ArousalLevel,
)
from .audio_analyzer import AudioAnalyzer, AudioAnalyzerConfig
from .lyric_analyzer import LyricAnalyzer, LyricAnalyzerConfig
from .psychoacoustic import PsychoacousticMapper, PsychoacousticConfig

logger = logging.getLogger(__name__)


@dataclass
class ResonanceSynthesizerConfig:
    """
    Configuration for resonance synthesizer.

    Attributes:
        audio_config: Audio analyzer configuration
        lyric_config: Lyric analyzer configuration
        psychoacoustic_config: Psychoacoustic mapper configuration
        generate_summary: Auto-generate impact summary
        generate_contexts: Auto-generate context recommendations
    """
    audio_config: Optional[AudioAnalyzerConfig] = None
    lyric_config: Optional[LyricAnalyzerConfig] = None
    psychoacoustic_config: Optional[PsychoacousticConfig] = None
    generate_summary: bool = True
    generate_contexts: bool = True


class ResonanceSynthesizer:
    """
    Main synthesizer for complete song resonance analysis.

    Combines audio analysis, lyric analysis, and psychoacoustic mapping
    into a comprehensive resonance profile.
    """

    def __init__(
        self,
        config: Optional[ResonanceSynthesizerConfig] = None,
    ):
        """
        Initialize synthesizer.

        Args:
            config: Synthesizer configuration
        """
        self.config = config or ResonanceSynthesizerConfig()

        self.audio_analyzer = AudioAnalyzer(self.config.audio_config)
        self.lyric_analyzer = LyricAnalyzer(self.config.lyric_config)
        self.psychoacoustic_mapper = PsychoacousticMapper(self.config.psychoacoustic_config)

    def analyze(
        self,
        audio: Optional[Union[np.ndarray, str, Path]] = None,
        lyrics: Optional[str] = None,
        sample_rate: Optional[int] = None,
        title: str = "",
        artist: str = "",
    ) -> ResonanceProfile:
        """
        Perform complete resonance analysis.

        Args:
            audio: Audio data (numpy array or path to file)
            lyrics: Lyrics text
            sample_rate: Sample rate if audio is array
            title: Song title
            artist: Artist name

        Returns:
            Complete resonance profile
        """
        profile = ResonanceProfile(
            song_title=title,
            artist=artist,
            analyzed_at=datetime.now(),
        )

        # Analyze audio
        if audio is not None:
            try:
                profile.audio_features = self.audio_analyzer.analyze(audio, sample_rate)
                logger.info(f"Audio analysis complete: {profile.audio_features.tempo_bpm:.0f} BPM")
            except Exception as e:
                logger.warning(f"Audio analysis failed: {e}")

        # Analyze lyrics
        if lyrics:
            try:
                profile.lyric_features = self.lyric_analyzer.analyze(lyrics)
                logger.info(f"Lyric analysis complete: sentiment={profile.lyric_features.sentiment_score:+.2f}")
            except Exception as e:
                logger.warning(f"Lyric analysis failed: {e}")

        # Map to psychological effects
        primary, secondary, cumulative = self.psychoacoustic_mapper.map_effects(
            profile.audio_features,
            profile.lyric_features,
        )
        profile.primary_effects = primary
        profile.secondary_effects = secondary
        profile.cumulative_effects = cumulative

        # Generate warnings
        profile.warnings = self.psychoacoustic_mapper.generate_warnings(
            profile.audio_features,
            profile.lyric_features,
        )

        # Calculate overall assessments
        profile.overall_valence = self._calculate_overall_valence(profile)
        profile.overall_arousal = self._calculate_overall_arousal(profile)
        profile.overall_intensity = self._calculate_overall_intensity(profile)

        # Generate summary and contexts
        if self.config.generate_summary:
            profile.listener_impact_summary = self._generate_summary(profile)

        if self.config.generate_contexts:
            profile.recommended_contexts = self._generate_recommended_contexts(profile)
            profile.contraindicated_contexts = self._generate_contraindicated_contexts(profile)

        return profile

    def analyze_file(
        self,
        audio_path: str,
        lyrics: Optional[str] = None,
        title: str = "",
        artist: str = "",
    ) -> ResonanceProfile:
        """
        Analyze audio file with optional lyrics.

        Args:
            audio_path: Path to audio file
            lyrics: Optional lyrics text
            title: Song title (auto-detected from filename if empty)
            artist: Artist name

        Returns:
            Complete resonance profile
        """
        if not title:
            # Extract from filename
            title = Path(audio_path).stem.replace("_", " ").replace("-", " ").title()

        return self.analyze(
            audio=audio_path,
            lyrics=lyrics,
            title=title,
            artist=artist,
        )

    def _calculate_overall_valence(self, profile: ResonanceProfile) -> EmotionalValence:
        """Calculate overall emotional valence."""
        valence_score = 0.0
        weight = 0.0

        # From audio
        if profile.audio_features:
            af = profile.audio_features
            # Major key contributes positive
            valence_score += (0.3 if af.is_major else -0.2)
            weight += 0.4

        # From lyrics
        if profile.lyric_features:
            valence_score += profile.lyric_features.sentiment_score * 0.6
            weight += 0.6

        if weight > 0:
            valence_score /= weight

        # Map to category
        if valence_score < -0.4:
            return EmotionalValence.VERY_NEGATIVE
        elif valence_score < -0.1:
            return EmotionalValence.NEGATIVE
        elif valence_score < 0.1:
            return EmotionalValence.NEUTRAL
        elif valence_score < 0.4:
            return EmotionalValence.POSITIVE
        else:
            return EmotionalValence.VERY_POSITIVE

    def _calculate_overall_arousal(self, profile: ResonanceProfile) -> ArousalLevel:
        """Calculate overall arousal level."""
        if profile.audio_features:
            return profile.audio_features.arousal_tendency
        return ArousalLevel.MODERATE

    def _calculate_overall_intensity(self, profile: ResonanceProfile) -> float:
        """Calculate overall emotional intensity (0-1)."""
        intensity = 0.5

        if profile.audio_features:
            af = profile.audio_features
            # Louder + more bass + more dynamic = more intense
            intensity += (af.loudness_db + 20) / 40 * 0.2
            intensity += af.bass_energy * 0.2
            intensity += af.dynamic_range_db / 30 * 0.1

        if profile.lyric_features:
            lf = profile.lyric_features
            # Strong sentiment (positive or negative) = more intense
            intensity += abs(lf.sentiment_score) * 0.3

        return min(1.0, max(0.0, intensity))

    def _generate_summary(self, profile: ResonanceProfile) -> str:
        """Generate human-readable impact summary."""
        parts = []

        # Opening
        valence_desc = {
            EmotionalValence.VERY_NEGATIVE: "deeply melancholic or dark",
            EmotionalValence.NEGATIVE: "somber or reflective",
            EmotionalValence.NEUTRAL: "emotionally balanced",
            EmotionalValence.POSITIVE: "uplifting or positive",
            EmotionalValence.VERY_POSITIVE: "joyful and energizing",
        }
        arousal_desc = {
            ArousalLevel.VERY_LOW: "deeply calming, potentially sedating",
            ArousalLevel.LOW: "relaxing and peaceful",
            ArousalLevel.MODERATE: "steady and grounded",
            ArousalLevel.HIGH: "energizing and activating",
            ArousalLevel.VERY_HIGH: "intense and potentially overwhelming",
        }

        parts.append(
            f"This song creates a {valence_desc[profile.overall_valence]} experience "
            f"that is {arousal_desc[profile.overall_arousal]}."
        )

        # Primary effects
        if profile.primary_effects:
            top_effects = profile.primary_effects[:2]
            effect_names = [e.name.lower() for e in top_effects]
            parts.append(
                f"Primary effects include {' and '.join(effect_names)}."
            )

        # Cumulative effects
        if profile.cumulative_effects:
            parts.append(
                "With repeated listening, listeners may experience: " +
                ", ".join(e.name.lower() for e in profile.cumulative_effects[:2]) + "."
            )

        # Warnings summary
        high_warnings = [w for w in profile.warnings if w.risk_level.value in ("high", "severe")]
        if high_warnings:
            parts.append(
                f"Note: This song contains elements that may be {high_warnings[0].category.lower()}."
            )

        return " ".join(parts)

    def _generate_recommended_contexts(self, profile: ResonanceProfile) -> List[str]:
        """Generate recommended listening contexts."""
        contexts = []

        valence = profile.overall_valence
        arousal = profile.overall_arousal

        # Based on valence + arousal combination
        if arousal in (ArousalLevel.VERY_LOW, ArousalLevel.LOW):
            contexts.append("Relaxation or meditation")
            contexts.append("Winding down before sleep")
            if valence in (EmotionalValence.POSITIVE, EmotionalValence.VERY_POSITIVE):
                contexts.append("Gentle morning routine")

        if arousal in (ArousalLevel.HIGH, ArousalLevel.VERY_HIGH):
            contexts.append("Exercise or workout")
            contexts.append("Getting energized")
            if valence in (EmotionalValence.POSITIVE, EmotionalValence.VERY_POSITIVE):
                contexts.append("Celebrations or parties")

        if valence in (EmotionalValence.NEGATIVE, EmotionalValence.VERY_NEGATIVE):
            contexts.append("Processing difficult emotions")
            contexts.append("Cathartic release")

        if valence in (EmotionalValence.POSITIVE, EmotionalValence.VERY_POSITIVE):
            contexts.append("Mood elevation")

        # Theme-based
        if profile.lyric_features:
            themes = profile.lyric_features.themes
            if themes.get("love", 0) > 0.3:
                contexts.append("Romantic moments")
            if themes.get("self-empowerment", 0) > 0.3:
                contexts.append("Building confidence")
            if themes.get("nostalgia", 0) > 0.3:
                contexts.append("Reminiscing")

        return contexts[:5]

    def _generate_contraindicated_contexts(self, profile: ResonanceProfile) -> List[str]:
        """Generate contexts where song may not be suitable."""
        contexts = []

        valence = profile.overall_valence
        arousal = profile.overall_arousal

        if arousal in (ArousalLevel.VERY_HIGH, ArousalLevel.HIGH):
            contexts.append("Trying to sleep or relax")
            contexts.append("Focused concentration work")

        if arousal in (ArousalLevel.VERY_LOW, ArousalLevel.LOW):
            contexts.append("Needing energy or motivation")

        if valence in (EmotionalValence.VERY_NEGATIVE, EmotionalValence.NEGATIVE):
            contexts.append("Already feeling depressed (may amplify)")
            contexts.append("Need for mood elevation")

        # Warning-based
        for warning in profile.warnings:
            if warning.risk_level in (RiskLevel.HIGH, RiskLevel.SEVERE):
                for group in warning.vulnerable_groups:
                    contexts.append(f"Listeners who are {group}")

        return contexts[:5]


def analyze_song(
    audio_path: Optional[str] = None,
    lyrics: Optional[str] = None,
    title: str = "",
    artist: str = "",
    config: Optional[ResonanceSynthesizerConfig] = None,
) -> ResonanceProfile:
    """
    Analyze a song for psychological resonance.

    Args:
        audio_path: Path to audio file
        lyrics: Lyrics text
        title: Song title
        artist: Artist name
        config: Optional configuration

    Returns:
        Complete resonance profile
    """
    synthesizer = ResonanceSynthesizer(config)

    if audio_path:
        return synthesizer.analyze_file(audio_path, lyrics, title, artist)
    else:
        return synthesizer.analyze(lyrics=lyrics, title=title, artist=artist)


def analyze_lyrics_only(
    lyrics: str,
    title: str = "",
    artist: str = "",
) -> ResonanceProfile:
    """
    Analyze lyrics only (no audio).

    Args:
        lyrics: Lyrics text
        title: Song title
        artist: Artist name

    Returns:
        Resonance profile based on lyrics
    """
    return analyze_song(lyrics=lyrics, title=title, artist=artist)


def generate_resonance_report(
    audio_path: Optional[str] = None,
    lyrics: Optional[str] = None,
    title: str = "",
    artist: str = "",
) -> str:
    """
    Generate markdown resonance report.

    Args:
        audio_path: Path to audio file
        lyrics: Lyrics text
        title: Song title
        artist: Artist name

    Returns:
        Markdown report
    """
    profile = analyze_song(audio_path, lyrics, title, artist)
    return profile.to_markdown()
