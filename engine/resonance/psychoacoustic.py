"""
VØX Resonance - Psychoacoustic Mapper
-------------------------------------

Maps audio and lyric features to predicted psychological effects.

Based on music psychology research:
    - Tempo → Arousal/heart rate
    - Mode (major/minor) → Valence
    - Bass → Physical/visceral response
    - Repetition → Memory embedding
    - Lyrics → Cognitive processing
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

from .models import (
    AudioFeatures,
    LyricFeatures,
    PsychologicalEffect,
    ResonanceCategory,
    ResonanceWarning,
    RiskLevel,
    ArousalLevel,
    EmotionalValence,
)

logger = logging.getLogger(__name__)


@dataclass
class PsychoacousticConfig:
    """
    Configuration for psychoacoustic mapping.

    Attributes:
        min_confidence: Minimum confidence to report effect
        include_warnings: Include risk warnings
        detailed_mechanisms: Include detailed explanations
    """
    min_confidence: float = 0.3
    include_warnings: bool = True
    detailed_mechanisms: bool = True


class PsychoacousticMapper:
    """
    Maps musical features to psychological effects.

    Uses established music psychology principles to predict
    how audio and lyric features will affect listeners.
    """

    def __init__(
        self,
        config: Optional[PsychoacousticConfig] = None,
    ):
        """
        Initialize mapper.

        Args:
            config: Mapper configuration
        """
        self.config = config or PsychoacousticConfig()

    def map_effects(
        self,
        audio: Optional[AudioFeatures],
        lyrics: Optional[LyricFeatures],
    ) -> Tuple[List[PsychologicalEffect], List[PsychologicalEffect], List[PsychologicalEffect]]:
        """
        Map features to psychological effects.

        Args:
            audio: Audio features
            lyrics: Lyric features

        Returns:
            Tuple of (primary_effects, secondary_effects, cumulative_effects)
        """
        primary = []
        secondary = []
        cumulative = []

        # Audio-based effects
        if audio:
            primary.extend(self._map_tempo_effects(audio))
            primary.extend(self._map_mode_effects(audio))
            primary.extend(self._map_spectral_effects(audio))
            secondary.extend(self._map_rhythm_effects(audio))
            secondary.extend(self._map_dynamic_effects(audio))

        # Lyric-based effects
        if lyrics:
            primary.extend(self._map_sentiment_effects(lyrics))
            secondary.extend(self._map_theme_effects(lyrics))
            secondary.extend(self._map_pronoun_effects(lyrics))
            cumulative.extend(self._map_repetition_effects(lyrics))

        # Combined effects
        if audio and lyrics:
            primary.extend(self._map_combined_effects(audio, lyrics))
            cumulative.extend(self._map_cumulative_combined(audio, lyrics))

        # Filter by confidence
        primary = [e for e in primary if e.confidence >= self.config.min_confidence]
        secondary = [e for e in secondary if e.confidence >= self.config.min_confidence]
        cumulative = [e for e in cumulative if e.confidence >= self.config.min_confidence]

        # Sort by intensity
        primary.sort(key=lambda x: -x.intensity)
        secondary.sort(key=lambda x: -x.intensity)
        cumulative.sort(key=lambda x: -x.intensity)

        return primary, secondary, cumulative

    def generate_warnings(
        self,
        audio: Optional[AudioFeatures],
        lyrics: Optional[LyricFeatures],
    ) -> List[ResonanceWarning]:
        """
        Generate warnings about potential negative effects.

        Args:
            audio: Audio features
            lyrics: Lyric features

        Returns:
            List of warnings
        """
        if not self.config.include_warnings:
            return []

        warnings = []

        # Audio warnings
        if audio:
            # Very loud/intense
            if audio.loudness_db > -6 and audio.bass_energy > 0.6:
                warnings.append(ResonanceWarning(
                    risk_level=RiskLevel.MODERATE,
                    category="Physiological Intensity",
                    description="High loudness with heavy bass may cause physical agitation or overstimulation.",
                    triggers=["high loudness", "heavy bass"],
                    vulnerable_groups=["anxiety-prone individuals", "sensory-sensitive listeners"],
                    recommendation="Listen at moderate volume, take breaks.",
                ))

            # High dissonance
            if audio.dissonance_score > 0.6:
                warnings.append(ResonanceWarning(
                    risk_level=RiskLevel.LOW,
                    category="Harmonic Tension",
                    description="High dissonance may create unresolved tension or discomfort.",
                    triggers=["dissonant harmonies"],
                    vulnerable_groups=["stress-sensitive individuals"],
                ))

        # Lyric warnings
        if lyrics:
            # Very negative sentiment
            if lyrics.sentiment_score < -0.5:
                warnings.append(ResonanceWarning(
                    risk_level=RiskLevel.MODERATE,
                    category="Negative Content",
                    description="Strongly negative lyrical content may reinforce negative thought patterns with repeated listening.",
                    triggers=["negative words", "dark themes"],
                    vulnerable_groups=["individuals with depression", "those in emotional distress"],
                    recommendation="Balance with positive content, be mindful of mood effects.",
                ))

            # Themes of self-harm or hopelessness
            if lyrics.themes.get("loss", 0) > 0.5 and lyrics.sentiment_score < -0.3:
                warnings.append(ResonanceWarning(
                    risk_level=RiskLevel.HIGH,
                    category="Loss/Grief Themes",
                    description="Strong themes of loss combined with negative sentiment may intensify grief or sadness.",
                    triggers=["loss themes", "negative sentiment"],
                    vulnerable_groups=["those experiencing loss", "individuals with depression"],
                    recommendation="May not be suitable during acute grief. Consider context of listening.",
                ))

            # High repetition of negative content
            if lyrics.repetition_score > 0.5 and lyrics.sentiment_score < -0.2:
                warnings.append(ResonanceWarning(
                    risk_level=RiskLevel.MODERATE,
                    category="Repetitive Negative Messaging",
                    description="Repeated negative phrases may become internalized with frequent listening.",
                    triggers=["repetition", "negative phrases"],
                    vulnerable_groups=["impressionable listeners", "those with negative self-talk patterns"],
                    recommendation="Be aware of subconscious messaging effects.",
                ))

        return warnings

    def _map_tempo_effects(self, audio: AudioFeatures) -> List[PsychologicalEffect]:
        """Map tempo to effects."""
        effects = []
        tempo = audio.tempo_bpm

        if tempo < 70:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.PHYSIOLOGICAL,
                name="Heart Rate Reduction",
                description="Slow tempo tends to reduce heart rate and breathing, inducing calm.",
                intensity=0.7,
                confidence=0.8,
                duration="transient",
                mechanism="Rhythmic entrainment - body syncs to slow external rhythm",
                evidence=["tempo < 70 BPM"],
            ))
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.EMOTIONAL,
                name="Relaxation Induction",
                description="Promotes a relaxed, contemplative state.",
                intensity=0.6,
                confidence=0.75,
                duration="short",
                mechanism="Parasympathetic nervous system activation",
                evidence=["slow tempo"],
            ))

        elif tempo > 120:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.PHYSIOLOGICAL,
                name="Arousal Increase",
                description="Fast tempo elevates heart rate and energy levels.",
                intensity=min(1.0, (tempo - 120) / 60),
                confidence=0.85,
                duration="transient",
                mechanism="Rhythmic entrainment - body syncs to fast external rhythm",
                evidence=[f"tempo = {tempo:.0f} BPM"],
            ))
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.BEHAVIORAL,
                name="Movement Activation",
                description="Encourages physical movement, dancing, exercise.",
                intensity=min(1.0, (tempo - 100) / 80),
                confidence=0.8,
                duration="transient",
                mechanism="Motor cortex activation via rhythm",
                evidence=["fast tempo"],
            ))

        return effects

    def _map_mode_effects(self, audio: AudioFeatures) -> List[PsychologicalEffect]:
        """Map major/minor mode to effects."""
        effects = []

        if audio.is_major:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.EMOTIONAL,
                name="Positive Valence Induction",
                description="Major key tends to evoke happiness, brightness, optimism.",
                intensity=0.6,
                confidence=0.75,
                duration="short",
                mechanism="Cultural and possibly innate association of major intervals with positive emotion",
                evidence=["major key"],
            ))
        else:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.EMOTIONAL,
                name="Melancholic/Serious Mood",
                description="Minor key tends to evoke sadness, seriousness, depth.",
                intensity=0.6,
                confidence=0.75,
                duration="short",
                mechanism="Cultural and possibly innate association of minor intervals with negative/complex emotion",
                evidence=["minor key"],
            ))

        return effects

    def _map_spectral_effects(self, audio: AudioFeatures) -> List[PsychologicalEffect]:
        """Map spectral characteristics to effects."""
        effects = []

        # Heavy bass
        if audio.bass_energy > 0.5:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.PHYSIOLOGICAL,
                name="Visceral/Embodied Response",
                description="Heavy bass frequencies are felt physically in the body.",
                intensity=audio.bass_energy,
                confidence=0.85,
                duration="transient",
                mechanism="Low frequencies physically vibrate the body, especially chest",
                evidence=[f"bass energy = {audio.bass_energy:.0%}"],
            ))

        # Bright/treble-heavy
        if audio.treble_energy > 0.5:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.COGNITIVE,
                name="Alertness Enhancement",
                description="Bright, treble-heavy sound promotes alertness and attention.",
                intensity=audio.treble_energy * 0.7,
                confidence=0.6,
                duration="transient",
                mechanism="High frequencies associated with alerting signals",
                evidence=[f"treble energy = {audio.treble_energy:.0%}"],
            ))

        return effects

    def _map_rhythm_effects(self, audio: AudioFeatures) -> List[PsychologicalEffect]:
        """Map rhythm characteristics to effects."""
        effects = []

        if audio.rhythm_regularity > 0.8:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.COGNITIVE,
                name="Predictability Comfort",
                description="Regular rhythm provides sense of stability and predictability.",
                intensity=0.5,
                confidence=0.7,
                duration="short",
                mechanism="Brain rewards predicted patterns with dopamine",
                evidence=[f"rhythm regularity = {audio.rhythm_regularity:.0%}"],
            ))
        elif audio.rhythm_regularity < 0.5:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.COGNITIVE,
                name="Attention Engagement",
                description="Irregular rhythm keeps the brain engaged and attentive.",
                intensity=0.6,
                confidence=0.65,
                duration="transient",
                mechanism="Unpredictability requires active processing",
                evidence=[f"rhythm regularity = {audio.rhythm_regularity:.0%}"],
            ))

        return effects

    def _map_dynamic_effects(self, audio: AudioFeatures) -> List[PsychologicalEffect]:
        """Map dynamic range to effects."""
        effects = []

        if audio.dynamic_range_db > 15:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.EMOTIONAL,
                name="Emotional Journey",
                description="Wide dynamic range creates emotional peaks and valleys.",
                intensity=0.7,
                confidence=0.7,
                duration="medium",
                mechanism="Contrast between loud and quiet creates tension and release",
                evidence=[f"dynamic range = {audio.dynamic_range_db:.1f} dB"],
            ))
        elif audio.dynamic_range_db < 6:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.COGNITIVE,
                name="Hypnotic/Trance Effect",
                description="Consistent loudness can induce trance-like or hypnotic state.",
                intensity=0.5,
                confidence=0.6,
                duration="medium",
                mechanism="Lack of dynamic change reduces conscious engagement",
                evidence=[f"dynamic range = {audio.dynamic_range_db:.1f} dB"],
            ))

        return effects

    def _map_sentiment_effects(self, lyrics: LyricFeatures) -> List[PsychologicalEffect]:
        """Map lyric sentiment to effects."""
        effects = []

        if lyrics.sentiment_score > 0.3:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.EMOTIONAL,
                name="Positive Mood Reinforcement",
                description="Positive lyrics reinforce and elevate positive emotions.",
                intensity=abs(lyrics.sentiment_score),
                confidence=0.75,
                duration="short",
                mechanism="Verbal priming of positive emotional states",
                evidence=[f"sentiment = {lyrics.sentiment_score:+.2f}"],
            ))
        elif lyrics.sentiment_score < -0.3:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.EMOTIONAL,
                name="Catharsis/Validation",
                description="Negative lyrics may provide catharsis or validation of difficult emotions.",
                intensity=abs(lyrics.sentiment_score),
                confidence=0.7,
                duration="short",
                mechanism="Emotional expression and validation through identification",
                evidence=[f"sentiment = {lyrics.sentiment_score:+.2f}"],
            ))

        return effects

    def _map_theme_effects(self, lyrics: LyricFeatures) -> List[PsychologicalEffect]:
        """Map themes to effects."""
        effects = []

        if lyrics.themes.get("love", 0) > 0.3:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.SOCIAL,
                name="Attachment Activation",
                description="Love themes activate attachment systems and relationship thoughts.",
                intensity=lyrics.themes["love"],
                confidence=0.7,
                duration="short",
                mechanism="Semantic priming of romantic/attachment schemas",
                evidence=["love theme detected"],
            ))

        if lyrics.themes.get("self-empowerment", 0) > 0.3:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.COGNITIVE,
                name="Self-Efficacy Boost",
                description="Empowerment themes may temporarily boost confidence and self-belief.",
                intensity=lyrics.themes["self-empowerment"],
                confidence=0.65,
                duration="short",
                mechanism="Verbal affirmation and identification with powerful narrative",
                evidence=["empowerment theme detected"],
            ))

        if lyrics.themes.get("nostalgia", 0) > 0.3:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.MEMORY,
                name="Autobiographical Memory Activation",
                description="Nostalgia themes trigger personal memories and bittersweet emotions.",
                intensity=lyrics.themes["nostalgia"],
                confidence=0.75,
                duration="medium",
                mechanism="Temporal self-reference activates episodic memory",
                evidence=["nostalgia theme detected"],
            ))

        return effects

    def _map_pronoun_effects(self, lyrics: LyricFeatures) -> List[PsychologicalEffect]:
        """Map pronoun usage to effects."""
        effects = []

        if lyrics.first_person_ratio > 0.1:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.COGNITIVE,
                name="Personal Identification",
                description="First-person lyrics encourage listener to adopt the singer's perspective.",
                intensity=min(1.0, lyrics.first_person_ratio * 5),
                confidence=0.7,
                duration="transient",
                mechanism="Self-referential processing via pronoun substitution",
                evidence=[f"first-person ratio = {lyrics.first_person_ratio:.0%}"],
            ))

        if lyrics.second_person_ratio > 0.08:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.SOCIAL,
                name="Direct Address Effect",
                description="Second-person 'you' creates sense of being spoken to directly.",
                intensity=min(1.0, lyrics.second_person_ratio * 6),
                confidence=0.7,
                duration="transient",
                mechanism="Direct address activates social processing",
                evidence=[f"second-person ratio = {lyrics.second_person_ratio:.0%}"],
            ))

        return effects

    def _map_repetition_effects(self, lyrics: LyricFeatures) -> List[PsychologicalEffect]:
        """Map repetition to cumulative effects."""
        effects = []

        if lyrics.repetition_score > 0.3:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.MEMORY,
                name="Message Embedding",
                description="Repeated phrases become embedded in memory and may influence thoughts.",
                intensity=lyrics.repetition_score,
                confidence=0.8,
                duration="lasting",
                mechanism="Repetition strengthens neural pathways (Hebbian learning)",
                evidence=[f"repetition score = {lyrics.repetition_score:.0%}",
                         f"repeated: {', '.join(lyrics.most_repeated[:2])}"],
            ))

            if lyrics.sentiment_score > 0.2:
                effects.append(PsychologicalEffect(
                    category=ResonanceCategory.COGNITIVE,
                    name="Positive Affirmation Effect",
                    description="Repeated positive messages may function as affirmations.",
                    intensity=lyrics.repetition_score * 0.8,
                    confidence=0.65,
                    duration="lasting",
                    mechanism="Repeated positive self-statements can influence self-concept",
                    evidence=["positive repetition"],
                ))

        return effects

    def _map_combined_effects(
        self,
        audio: AudioFeatures,
        lyrics: LyricFeatures,
    ) -> List[PsychologicalEffect]:
        """Map combined audio+lyric effects."""
        effects = []

        # Congruent valence (audio and lyrics match)
        audio_positive = audio.is_major and audio.tempo_bpm > 100
        lyric_positive = lyrics.sentiment_score > 0.2

        if audio_positive and lyric_positive:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.EMOTIONAL,
                name="Amplified Positivity",
                description="Upbeat music + positive lyrics create reinforced positive emotional state.",
                intensity=0.8,
                confidence=0.8,
                duration="short",
                mechanism="Multi-channel positive emotional priming",
                evidence=["major key", "fast tempo", "positive lyrics"],
            ))

        audio_negative = not audio.is_major and audio.tempo_bpm < 90
        lyric_negative = lyrics.sentiment_score < -0.2

        if audio_negative and lyric_negative:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.EMOTIONAL,
                name="Amplified Melancholy",
                description="Slow/minor music + negative lyrics create deep emotional resonance.",
                intensity=0.8,
                confidence=0.8,
                duration="medium",
                mechanism="Multi-channel negative/complex emotional priming",
                evidence=["minor key", "slow tempo", "negative lyrics"],
            ))

        # Incongruent (mixed signals)
        if audio_positive and lyric_negative:
            effects.append(PsychologicalEffect(
                category=ResonanceCategory.COGNITIVE,
                name="Ironic/Complex Response",
                description="Upbeat music with sad lyrics creates complex, ironic emotional experience.",
                intensity=0.6,
                confidence=0.65,
                duration="short",
                mechanism="Cognitive dissonance between channels creates interest/complexity",
                evidence=["positive music", "negative lyrics"],
            ))

        return effects

    def _map_cumulative_combined(
        self,
        audio: AudioFeatures,
        lyrics: LyricFeatures,
    ) -> List[PsychologicalEffect]:
        """Map cumulative combined effects."""
        effects = []

        # High repetition + strong emotion + bass
        if (lyrics.repetition_score > 0.4 and
            abs(lyrics.sentiment_score) > 0.3 and
            audio.bass_energy > 0.5):

            effects.append(PsychologicalEffect(
                category=ResonanceCategory.BEHAVIORAL,
                name="Motivational Conditioning",
                description="Repeated emotional message with physical impact may condition behavioral associations.",
                intensity=0.7,
                confidence=0.6,
                duration="lasting",
                mechanism="Classical conditioning via repeated multi-sensory emotional pairing",
                evidence=["high repetition", "strong emotion", "physical bass"],
            ))

        return effects


def map_psychoacoustic_effects(
    audio: Optional[AudioFeatures],
    lyrics: Optional[LyricFeatures],
    config: Optional[PsychoacousticConfig] = None,
) -> Tuple[List[PsychologicalEffect], List[ResonanceWarning]]:
    """
    Map features to psychological effects and warnings.

    Args:
        audio: Audio features
        lyrics: Lyric features
        config: Optional mapper config

    Returns:
        Tuple of (all_effects, warnings)
    """
    mapper = PsychoacousticMapper(config)

    primary, secondary, cumulative = mapper.map_effects(audio, lyrics)
    warnings = mapper.generate_warnings(audio, lyrics)

    all_effects = primary + secondary + cumulative
    return all_effects, warnings
