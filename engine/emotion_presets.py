"""
AXIOM VOX Emotion Presets
-------------------------

Named emotion configurations for expressive TTS.

Presets map to the 5-dimensional emotional space:
- Valence: -1 (negative) to +1 (positive)
- Arousal: 0 (calm) to 1 (excited)
- Dominance: 0 (submissive) to 1 (dominant)
- Warmth: 0 (cold) to 1 (warm)
- Confidence: 0 (uncertain) to 1 (confident)

Usage:
    from axiom_vox.emotion_presets import get_emotion_preset, EMOTION_PRESETS

    preset = get_emotion_preset("joy")
    intent = create_intent_from_preset("empathetic")
"""

from enum import Enum
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict


class EmotionPresetName(str, Enum):
    """Named emotion presets."""
    # Basic emotions (Ekman's universal emotions)
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"

    # Complex/social emotions
    NEUTRAL = "neutral"
    CALM = "calm"
    EXCITED = "excited"
    PROFESSIONAL = "professional"
    WARM = "warm"
    AUTHORITATIVE = "authoritative"
    URGENT = "urgent"
    EMPATHETIC = "empathetic"
    CONFIDENT = "confident"
    APOLOGETIC = "apologetic"
    REASSURING = "reassuring"
    CONTEMPLATIVE = "contemplative"


@dataclass
class EmotionPreset:
    """
    Full emotion preset configuration.

    Maps a named emotion to the 5D emotional space used by
    EmotionalIntent and prosody control systems.
    """
    name: EmotionPresetName

    # 5D emotional space (matches EmotionalIntent)
    valence: float      # -1 (negative) to +1 (positive)
    arousal: float      # 0 (calm) to 1 (excited)
    dominance: float    # 0 (submissive) to 1 (dominant)
    warmth: float       # 0 (cold) to 1 (warm)
    confidence: float   # 0 (uncertain) to 1 (confident)

    # Prosody hints
    speaking_rate: float = 1.0   # 0.5 to 2.0
    pitch_base: float = 0.0      # Semitones from neutral
    pitch_variance: float = 0.5  # 0 (monotone) to 1 (varied)

    # Metadata
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name.value,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "warmth": self.warmth,
            "confidence": self.confidence,
            "speaking_rate": self.speaking_rate,
            "pitch_base": self.pitch_base,
            "pitch_variance": self.pitch_variance,
            "description": self.description,
        }


# ============================================================================
# PRESET REGISTRY
# ============================================================================

EMOTION_PRESETS: Dict[str, EmotionPreset] = {
    # Basic emotions
    "joy": EmotionPreset(
        name=EmotionPresetName.JOY,
        valence=0.8, arousal=0.7, dominance=0.5, warmth=0.8, confidence=0.7,
        speaking_rate=1.05, pitch_base=1.0, pitch_variance=0.7,
        description="Bright, happy, uplifting tone with natural energy"
    ),

    "sadness": EmotionPreset(
        name=EmotionPresetName.SADNESS,
        valence=-0.7, arousal=0.2, dominance=0.3, warmth=0.4, confidence=0.3,
        speaking_rate=0.85, pitch_base=-1.0, pitch_variance=0.3,
        description="Somber, subdued, reflective tone with slower pacing"
    ),

    "anger": EmotionPreset(
        name=EmotionPresetName.ANGER,
        valence=-0.6, arousal=0.9, dominance=0.8, warmth=0.1, confidence=0.9,
        speaking_rate=1.1, pitch_base=0.5, pitch_variance=0.8,
        description="Intense, forceful tone with increased energy"
    ),

    "fear": EmotionPreset(
        name=EmotionPresetName.FEAR,
        valence=-0.8, arousal=0.8, dominance=0.2, warmth=0.2, confidence=0.2,
        speaking_rate=1.15, pitch_base=1.5, pitch_variance=0.9,
        description="Tense, anxious tone with elevated pitch"
    ),

    "surprise": EmotionPreset(
        name=EmotionPresetName.SURPRISE,
        valence=0.3, arousal=0.8, dominance=0.4, warmth=0.5, confidence=0.4,
        speaking_rate=1.1, pitch_base=2.0, pitch_variance=0.9,
        description="Startled, astonished tone with pitch excursion"
    ),

    "disgust": EmotionPreset(
        name=EmotionPresetName.DISGUST,
        valence=-0.7, arousal=0.5, dominance=0.6, warmth=0.1, confidence=0.7,
        speaking_rate=0.95, pitch_base=-0.5, pitch_variance=0.4,
        description="Disapproving, averse tone"
    ),

    # Complex/social emotions
    "neutral": EmotionPreset(
        name=EmotionPresetName.NEUTRAL,
        valence=0.0, arousal=0.4, dominance=0.5, warmth=0.5, confidence=0.5,
        speaking_rate=1.0, pitch_base=0.0, pitch_variance=0.5,
        description="Balanced, matter-of-fact tone without strong emotion"
    ),

    "calm": EmotionPreset(
        name=EmotionPresetName.CALM,
        valence=0.3, arousal=0.2, dominance=0.5, warmth=0.7, confidence=0.6,
        speaking_rate=0.9, pitch_base=-0.5, pitch_variance=0.3,
        description="Peaceful, relaxed, soothing tone"
    ),

    "excited": EmotionPreset(
        name=EmotionPresetName.EXCITED,
        valence=0.7, arousal=0.9, dominance=0.6, warmth=0.6, confidence=0.8,
        speaking_rate=1.2, pitch_base=1.5, pitch_variance=0.9,
        description="Enthusiastic, energetic, animated tone"
    ),

    "professional": EmotionPreset(
        name=EmotionPresetName.PROFESSIONAL,
        valence=0.2, arousal=0.4, dominance=0.6, warmth=0.5, confidence=0.8,
        speaking_rate=1.0, pitch_base=0.0, pitch_variance=0.4,
        description="Clear, business-appropriate, competent tone"
    ),

    "warm": EmotionPreset(
        name=EmotionPresetName.WARM,
        valence=0.5, arousal=0.4, dominance=0.4, warmth=0.9, confidence=0.6,
        speaking_rate=0.95, pitch_base=0.5, pitch_variance=0.5,
        description="Friendly, caring, approachable tone"
    ),

    "authoritative": EmotionPreset(
        name=EmotionPresetName.AUTHORITATIVE,
        valence=0.1, arousal=0.5, dominance=0.9, warmth=0.3, confidence=0.9,
        speaking_rate=0.95, pitch_base=-1.0, pitch_variance=0.3,
        description="Commanding, expert, decisive tone"
    ),

    "urgent": EmotionPreset(
        name=EmotionPresetName.URGENT,
        valence=-0.1, arousal=0.8, dominance=0.7, warmth=0.3, confidence=0.8,
        speaking_rate=1.15, pitch_base=0.5, pitch_variance=0.6,
        description="Pressing, time-sensitive, insistent tone"
    ),

    "empathetic": EmotionPreset(
        name=EmotionPresetName.EMPATHETIC,
        valence=0.3, arousal=0.3, dominance=0.3, warmth=0.9, confidence=0.5,
        speaking_rate=0.9, pitch_base=0.0, pitch_variance=0.5,
        description="Understanding, compassionate, supportive tone"
    ),

    "confident": EmotionPreset(
        name=EmotionPresetName.CONFIDENT,
        valence=0.4, arousal=0.5, dominance=0.8, warmth=0.6, confidence=0.9,
        speaking_rate=1.0, pitch_base=-0.5, pitch_variance=0.4,
        description="Self-assured, certain, assertive tone"
    ),

    "apologetic": EmotionPreset(
        name=EmotionPresetName.APOLOGETIC,
        valence=-0.2, arousal=0.3, dominance=0.2, warmth=0.7, confidence=0.3,
        speaking_rate=0.9, pitch_base=0.5, pitch_variance=0.5,
        description="Regretful, humble, sincere tone"
    ),

    "reassuring": EmotionPreset(
        name=EmotionPresetName.REASSURING,
        valence=0.4, arousal=0.3, dominance=0.5, warmth=0.8, confidence=0.7,
        speaking_rate=0.9, pitch_base=0.0, pitch_variance=0.4,
        description="Comforting, calming, supportive tone"
    ),

    "contemplative": EmotionPreset(
        name=EmotionPresetName.CONTEMPLATIVE,
        valence=0.1, arousal=0.2, dominance=0.4, warmth=0.5, confidence=0.5,
        speaking_rate=0.85, pitch_base=-0.5, pitch_variance=0.3,
        description="Thoughtful, reflective, measured tone"
    ),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_emotion_preset(name: str) -> EmotionPreset:
    """
    Get an emotion preset by name.

    Args:
        name: Preset name (e.g., "joy", "empathetic")

    Returns:
        EmotionPreset instance

    Raises:
        KeyError: If preset name not found
    """
    name_lower = name.lower()
    if name_lower not in EMOTION_PRESETS:
        available = ", ".join(sorted(EMOTION_PRESETS.keys()))
        raise KeyError(f"Unknown emotion preset: '{name}'. Available: {available}")
    return EMOTION_PRESETS[name_lower]


def create_intent_from_preset(name: str) -> "EmotionalIntent":
    """
    Create an EmotionalIntent from a named preset.

    Args:
        name: Preset name

    Returns:
        EmotionalIntent instance with preset values
    """
    from axiom_vox.prosody_guardrails import EmotionalIntent

    preset = get_emotion_preset(name)
    return EmotionalIntent(
        valence=preset.valence,
        arousal=preset.arousal,
        dominance=preset.dominance,
        warmth=preset.warmth,
        confidence=preset.confidence,
        target_emotion=preset.name.value,
        speaking_rate=preset.speaking_rate,
        pitch_variation=preset.pitch_variance,
    )


def list_emotion_presets() -> Dict[str, Dict[str, Any]]:
    """
    List all available emotion presets.

    Returns:
        Dictionary mapping preset names to their configurations
    """
    return {name: preset.to_dict() for name, preset in EMOTION_PRESETS.items()}


def validate_preset_name(name: str) -> bool:
    """
    Check if a preset name is valid.

    Args:
        name: Preset name to validate

    Returns:
        True if valid, False otherwise
    """
    return name.lower() in EMOTION_PRESETS


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  AXIOM VOX Emotion Presets")
    print("=" * 70)

    print("\nAvailable presets:\n")
    print(f"{'Preset':<15} {'Valence':>8} {'Arousal':>8} {'Dominance':>10} {'Warmth':>8} {'Rate':>6}")
    print("-" * 70)

    for name, preset in sorted(EMOTION_PRESETS.items()):
        print(f"{name:<15} {preset.valence:>8.1f} {preset.arousal:>8.1f} "
              f"{preset.dominance:>10.1f} {preset.warmth:>8.1f} {preset.speaking_rate:>6.2f}")

    print("\n" + "-" * 70)
    print("\nTesting preset lookup:\n")

    for test_name in ["joy", "empathetic", "professional"]:
        preset = get_emotion_preset(test_name)
        print(f"  {test_name}: {preset.description}")

    print("\nTesting EmotionalIntent creation:\n")

    try:
        intent = create_intent_from_preset("warm")
        print(f"  warm -> EmotionalIntent(valence={intent.valence}, warmth={intent.warmth})")
    except ImportError:
        print("  (EmotionalIntent not available - prosody_guardrails not imported)")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
