"""
VØX Resonance Analysis
----------------------

Analyze songs (audio + lyrics) to predict psychological effects on listeners.

Features:
    - Audio feature extraction (tempo, key, spectrum, dynamics)
    - Lyric analysis (sentiment, themes, repetition, imagery)
    - Psychoacoustic mapping (features → psychological effects)
    - Impact prediction (what the song will do to listeners)
    - Risk warnings (potential negative effects)

Quick Start:
    >>> from axiom_vox.resonance import analyze_song, generate_resonance_report
    >>>
    >>> # Analyze a song
    >>> profile = analyze_song(
    ...     audio_path="song.mp3",
    ...     lyrics="I feel the music in my soul...",
    ...     title="My Song",
    ...     artist="Artist Name",
    ... )
    >>>
    >>> # Get the impact summary
    >>> print(profile.listener_impact_summary)
    >>>
    >>> # Get detailed effects
    >>> for effect in profile.primary_effects:
    ...     print(f"{effect.name}: {effect.description}")
    >>>
    >>> # Generate full report
    >>> report = generate_resonance_report(audio_path="song.mp3", lyrics="...")
    >>> print(report)

Usage with lyrics only:
    >>> from axiom_vox.resonance import analyze_lyrics_only
    >>>
    >>> profile = analyze_lyrics_only('''
    ...     I will survive, I will be strong
    ...     Nothing can bring me down
    ... ''')
    >>> print(profile.overall_valence)  # POSITIVE
"""

from .models import (
    # Enums
    MusicalKey,
    EmotionalValence,
    ArousalLevel,
    ResonanceCategory,
    RiskLevel,
    # Feature models
    AudioFeatures,
    LyricFeatures,
    # Effect models
    PsychologicalEffect,
    ResonanceWarning,
    # Main profile
    ResonanceProfile,
)

from .audio_analyzer import (
    AudioAnalyzerConfig,
    AudioAnalyzer,
    analyze_audio,
)

from .lyric_analyzer import (
    LyricAnalyzerConfig,
    LyricAnalyzer,
    analyze_lyrics,
)

from .psychoacoustic import (
    PsychoacousticConfig,
    PsychoacousticMapper,
    map_psychoacoustic_effects,
)

from .synthesizer import (
    ResonanceSynthesizerConfig,
    ResonanceSynthesizer,
    analyze_song,
    analyze_lyrics_only,
    generate_resonance_report,
)


__all__ = [
    # Enums
    "MusicalKey",
    "EmotionalValence",
    "ArousalLevel",
    "ResonanceCategory",
    "RiskLevel",
    # Feature models
    "AudioFeatures",
    "LyricFeatures",
    # Effect models
    "PsychologicalEffect",
    "ResonanceWarning",
    # Main profile
    "ResonanceProfile",
    # Audio analyzer
    "AudioAnalyzerConfig",
    "AudioAnalyzer",
    "analyze_audio",
    # Lyric analyzer
    "LyricAnalyzerConfig",
    "LyricAnalyzer",
    "analyze_lyrics",
    # Psychoacoustic mapper
    "PsychoacousticConfig",
    "PsychoacousticMapper",
    "map_psychoacoustic_effects",
    # Synthesizer
    "ResonanceSynthesizerConfig",
    "ResonanceSynthesizer",
    "analyze_song",
    "analyze_lyrics_only",
    "generate_resonance_report",
]
