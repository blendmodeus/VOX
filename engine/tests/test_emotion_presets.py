"""
Tests for Emotion Presets
-------------------------

Tests for:
- EmotionPreset: Data structure
- EMOTION_PRESETS: Preset registry
- get_emotion_preset: Lookup function
- create_intent_from_preset: EmotionalIntent creation
- Integration with prosody system
"""

import pytest


# ============================================================================
# EMOTION PRESET TESTS
# ============================================================================

class TestEmotionPreset:
    """Tests for EmotionPreset dataclass."""

    def test_preset_creation(self):
        """Test creating an emotion preset."""
        from axiom_vox.emotion_presets import EmotionPreset, EmotionPresetName

        preset = EmotionPreset(
            name=EmotionPresetName.JOY,
            valence=0.8,
            arousal=0.7,
            dominance=0.5,
            warmth=0.8,
            confidence=0.7,
            speaking_rate=1.05,
            description="Happy preset",
        )

        assert preset.name == EmotionPresetName.JOY
        assert preset.valence == 0.8
        assert preset.arousal == 0.7
        assert preset.speaking_rate == 1.05

    def test_preset_to_dict(self):
        """Test preset serialization."""
        from axiom_vox.emotion_presets import EmotionPreset, EmotionPresetName

        preset = EmotionPreset(
            name=EmotionPresetName.CALM,
            valence=0.3,
            arousal=0.2,
            dominance=0.5,
            warmth=0.7,
            confidence=0.6,
        )

        d = preset.to_dict()

        assert d["name"] == "calm"
        assert d["valence"] == 0.3
        assert d["arousal"] == 0.2


# ============================================================================
# PRESET REGISTRY TESTS
# ============================================================================

class TestEmotionPresets:
    """Tests for EMOTION_PRESETS registry."""

    def test_all_presets_exist(self):
        """Test that all expected presets are defined."""
        from axiom_vox.emotion_presets import EMOTION_PRESETS

        expected_presets = [
            "joy", "sadness", "anger", "fear", "surprise", "disgust",
            "neutral", "calm", "excited", "professional", "warm",
            "authoritative", "urgent", "empathetic", "confident",
            "apologetic", "reassuring", "contemplative",
        ]

        for name in expected_presets:
            assert name in EMOTION_PRESETS, f"Missing preset: {name}"

    def test_all_presets_have_valid_values(self):
        """Test that all presets have valid value ranges."""
        from axiom_vox.emotion_presets import EMOTION_PRESETS

        for name, preset in EMOTION_PRESETS.items():
            # Valence: -1 to 1
            assert -1 <= preset.valence <= 1, f"{name}: invalid valence"
            # Arousal: 0 to 1
            assert 0 <= preset.arousal <= 1, f"{name}: invalid arousal"
            # Dominance: 0 to 1
            assert 0 <= preset.dominance <= 1, f"{name}: invalid dominance"
            # Warmth: 0 to 1
            assert 0 <= preset.warmth <= 1, f"{name}: invalid warmth"
            # Confidence: 0 to 1
            assert 0 <= preset.confidence <= 1, f"{name}: invalid confidence"
            # Speaking rate: 0.5 to 2.0
            assert 0.5 <= preset.speaking_rate <= 2.0, f"{name}: invalid speaking_rate"

    def test_joy_preset_values(self):
        """Test specific values for joy preset."""
        from axiom_vox.emotion_presets import EMOTION_PRESETS

        joy = EMOTION_PRESETS["joy"]

        # Joy should be positive, energetic, warm
        assert joy.valence > 0.5
        assert joy.arousal > 0.5
        assert joy.warmth > 0.5

    def test_sadness_preset_values(self):
        """Test specific values for sadness preset."""
        from axiom_vox.emotion_presets import EMOTION_PRESETS

        sadness = EMOTION_PRESETS["sadness"]

        # Sadness should be negative, low energy, slower
        assert sadness.valence < 0
        assert sadness.arousal < 0.5
        assert sadness.speaking_rate < 1.0

    def test_calm_preset_values(self):
        """Test specific values for calm preset."""
        from axiom_vox.emotion_presets import EMOTION_PRESETS

        calm = EMOTION_PRESETS["calm"]

        # Calm should be low arousal, warm, slower
        assert calm.arousal < 0.5
        assert calm.warmth > 0.5
        assert calm.speaking_rate <= 1.0

    def test_authoritative_preset_values(self):
        """Test specific values for authoritative preset."""
        from axiom_vox.emotion_presets import EMOTION_PRESETS

        auth = EMOTION_PRESETS["authoritative"]

        # Authoritative should be high dominance, high confidence
        assert auth.dominance > 0.7
        assert auth.confidence > 0.7


# ============================================================================
# LOOKUP FUNCTION TESTS
# ============================================================================

class TestGetEmotionPreset:
    """Tests for get_emotion_preset function."""

    def test_get_valid_preset(self):
        """Test getting a valid preset."""
        from axiom_vox.emotion_presets import get_emotion_preset

        preset = get_emotion_preset("joy")

        assert preset is not None
        assert preset.valence == 0.8

    def test_get_preset_case_insensitive(self):
        """Test that preset lookup is case-insensitive."""
        from axiom_vox.emotion_presets import get_emotion_preset

        preset_lower = get_emotion_preset("joy")
        preset_upper = get_emotion_preset("JOY")
        preset_mixed = get_emotion_preset("Joy")

        assert preset_lower.valence == preset_upper.valence == preset_mixed.valence

    def test_get_invalid_preset_raises(self):
        """Test that invalid preset name raises KeyError."""
        from axiom_vox.emotion_presets import get_emotion_preset

        with pytest.raises(KeyError) as exc_info:
            get_emotion_preset("nonexistent_emotion")

        assert "nonexistent_emotion" in str(exc_info.value)


# ============================================================================
# EMOTIONAL INTENT CREATION TESTS
# ============================================================================

class TestCreateIntentFromPreset:
    """Tests for create_intent_from_preset function."""

    def test_create_intent_from_joy(self):
        """Test creating EmotionalIntent from joy preset."""
        from axiom_vox.emotion_presets import create_intent_from_preset, EMOTION_PRESETS

        intent = create_intent_from_preset("joy")
        preset = EMOTION_PRESETS["joy"]

        assert intent.valence == preset.valence
        assert intent.arousal == preset.arousal
        assert intent.dominance == preset.dominance
        assert intent.warmth == preset.warmth
        assert intent.confidence == preset.confidence

    def test_create_intent_preserves_target_emotion(self):
        """Test that target_emotion is set correctly."""
        from axiom_vox.emotion_presets import create_intent_from_preset

        intent = create_intent_from_preset("empathetic")

        assert intent.target_emotion == "empathetic"

    def test_create_intent_includes_speaking_rate(self):
        """Test that speaking_rate is preserved."""
        from axiom_vox.emotion_presets import create_intent_from_preset, EMOTION_PRESETS

        intent = create_intent_from_preset("sadness")
        preset = EMOTION_PRESETS["sadness"]

        assert intent.speaking_rate == preset.speaking_rate

    def test_create_intent_invalid_raises(self):
        """Test that invalid preset raises error."""
        from axiom_vox.emotion_presets import create_intent_from_preset

        with pytest.raises((KeyError, Exception)):
            create_intent_from_preset("not_a_real_emotion")


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_list_emotion_presets(self):
        """Test listing all presets."""
        from axiom_vox.emotion_presets import list_emotion_presets

        presets = list_emotion_presets()

        assert isinstance(presets, dict)
        assert "joy" in presets
        assert "sadness" in presets
        assert "valence" in presets["joy"]

    def test_validate_preset_name(self):
        """Test preset name validation."""
        from axiom_vox.emotion_presets import validate_preset_name

        assert validate_preset_name("joy") == True
        assert validate_preset_name("JOY") == True
        assert validate_preset_name("invalid") == False
        assert validate_preset_name("") == False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEmotionPresetsIntegration:
    """Integration tests with prosody system."""

    def test_preset_to_prosody_decision(self):
        """Test using preset with ProsodyGuardrails."""
        from axiom_vox.emotion_presets import create_intent_from_preset
        from axiom_vox.prosody_guardrails import ProsodyGuardrails

        guardrails = ProsodyGuardrails()
        intent = create_intent_from_preset("professional")

        decision = guardrails.govern(
            text="Welcome to our quarterly report.",
            emotional_intent=intent,
            context={"domain": "finance"},
        )

        assert decision.approved == True

    def test_high_arousal_preset_triggers_check(self):
        """Test that high-arousal preset is checked."""
        from axiom_vox.emotion_presets import create_intent_from_preset
        from axiom_vox.prosody_guardrails import ProsodyGuardrails

        guardrails = ProsodyGuardrails(strict_mode=True)
        intent = create_intent_from_preset("excited")

        # Excited has high arousal - should be checked with urgency content
        decision = guardrails.govern(
            text="Buy now! Limited time only!",
            emotional_intent=intent,
        )

        # Should detect potential manipulation
        assert decision.manipulation_detected == True

    def test_empathetic_preset_in_medical_domain(self):
        """Test empathetic preset in medical context."""
        from axiom_vox.emotion_presets import create_intent_from_preset
        from axiom_vox.prosody_guardrails import ProsodyGuardrails

        guardrails = ProsodyGuardrails()
        intent = create_intent_from_preset("empathetic")

        decision = guardrails.govern(
            text="I understand this is difficult news.",
            emotional_intent=intent,
            context={"domain": "medical"},
        )

        # Empathetic should work well in medical context
        assert decision.approved == True


# ============================================================================
# ENUM TESTS
# ============================================================================

class TestEmotionPresetName:
    """Tests for EmotionPresetName enum."""

    def test_enum_values(self):
        """Test enum value strings."""
        from axiom_vox.emotion_presets import EmotionPresetName

        assert EmotionPresetName.JOY.value == "joy"
        assert EmotionPresetName.SADNESS.value == "sadness"
        assert EmotionPresetName.PROFESSIONAL.value == "professional"

    def test_enum_from_string(self):
        """Test creating enum from string."""
        from axiom_vox.emotion_presets import EmotionPresetName

        assert EmotionPresetName("joy") == EmotionPresetName.JOY
        assert EmotionPresetName("calm") == EmotionPresetName.CALM


# ============================================================================
# MODULE EXPORTS TEST
# ============================================================================

class TestEmotionPresetsExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """Test that all public items are exported."""
        from axiom_vox.emotion_presets import (
            EmotionPreset,
            EmotionPresetName,
            EMOTION_PRESETS,
            get_emotion_preset,
            create_intent_from_preset,
            list_emotion_presets,
            validate_preset_name,
        )

        assert EmotionPreset is not None
        assert EmotionPresetName is not None
        assert isinstance(EMOTION_PRESETS, dict)
        assert callable(get_emotion_preset)
        assert callable(create_intent_from_preset)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
