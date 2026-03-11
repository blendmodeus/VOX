"""
VØX Voice Genome
----------------

Complete voice DNA mapping and analysis.

The Voice Genome reveals everything about a voice:
- Biometric: Physical/health markers (age, fatigue, hydration, respiratory)
- Psychometric: Mental/emotional state (stress, authenticity, emotions, personality)
- Sociometric: Social dynamics (authority, warmth, trust, persuasion, charisma)

Quick Start:
    >>> from axiom_vox.genome import analyze_voice_genome, generate_genome_report
    >>>
    >>> # Analyze a voice
    >>> genome = analyze_voice_genome("voice_sample.wav")
    >>>
    >>> # Get overall scores
    >>> print(f"Health: {genome.voice_health_score:.0%}")
    >>> print(f"Authenticity: {genome.voice_authenticity_score:.0%}")
    >>> print(f"Influence: {genome.voice_influence_score:.0%}")
    >>>
    >>> # Get detailed biometric info
    >>> print(f"Estimated Age: {genome.biometric.estimated_age:.0f}")
    >>> print(f"Stress Level: {genome.psychometric.stress_level:.0%}")
    >>> print(f"Charisma: {genome.sociometric.charisma_score:.0%}")
    >>>
    >>> # Generate full report
    >>> report = generate_genome_report("voice_sample.wav")
    >>> print(report)

Voice Genome reveals the complete DNA of a voice - health, psychology, and social influence.
"""

from .models import (
    # Enums
    HealthRisk,
    ConfidenceLevel,
    EmotionalState,
    AuthenticityLevel,
    DominanceLevel,
    # Feature models
    AcousticFeatures,
    # Marker models
    BiometricMarkers,
    PsychometricMarkers,
    SociometricMarkers,
    # Main genome
    VoiceGenome,
    GenomeComparison,
)

from .extractor import (
    ExtractorConfig,
    VoiceFeatureExtractor,
    extract_voice_features,
)

from .biometric import (
    BiometricConfig,
    BiometricAnalyzer,
    analyze_biometrics,
)

from .psychometric import (
    PsychometricConfig,
    PsychometricAnalyzer,
    analyze_psychometrics,
)

from .sociometric import (
    SociometricConfig,
    SociometricAnalyzer,
    analyze_sociometrics,
)

from .synthesizer import (
    GenomeSynthesizerConfig,
    VoiceGenomeSynthesizer,
    analyze_voice_genome,
    generate_genome_report,
)


__all__ = [
    # Enums
    "HealthRisk",
    "ConfidenceLevel",
    "EmotionalState",
    "AuthenticityLevel",
    "DominanceLevel",
    # Feature models
    "AcousticFeatures",
    # Marker models
    "BiometricMarkers",
    "PsychometricMarkers",
    "SociometricMarkers",
    # Main genome
    "VoiceGenome",
    "GenomeComparison",
    # Extractor
    "ExtractorConfig",
    "VoiceFeatureExtractor",
    "extract_voice_features",
    # Biometric
    "BiometricConfig",
    "BiometricAnalyzer",
    "analyze_biometrics",
    # Psychometric
    "PsychometricConfig",
    "PsychometricAnalyzer",
    "analyze_psychometrics",
    # Sociometric
    "SociometricConfig",
    "SociometricAnalyzer",
    "analyze_sociometrics",
    # Synthesizer
    "GenomeSynthesizerConfig",
    "VoiceGenomeSynthesizer",
    "analyze_voice_genome",
    "generate_genome_report",
]
