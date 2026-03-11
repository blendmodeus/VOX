"""
Tests for AXIOM VOX Multi-Voice Synthesis (v0.9.0)

Tests cover:
- DialogueLine and DialogueScript data models
- CharacterRegistry functionality
- SSML <voice> tag parsing
- TransitionProcessor audio operations
- MultiVoiceSynthesizer integration

Run with: pytest axiom_vox/tests/test_multi_voice.py -v
"""

import pytest
import struct
from dataclasses import asdict
from typing import Set


# ============================================================================
# MULTI-VOICE DATA MODEL TESTS
# ============================================================================

class TestDialogueLine:
    """Tests for DialogueLine dataclass."""

    def test_basic_creation(self):
        """Test creating a DialogueLine with minimal args."""
        from axiom_vox.multi_voice import DialogueLine

        line = DialogueLine(text="Hello", voice_id="professional")
        assert line.text == "Hello"
        assert line.voice_id == "professional"
        assert line.character_name is None
        assert line.emotion is None
        assert line.pause_before_ms == 0
        assert line.pause_after_ms == 0

    def test_full_creation(self):
        """Test creating a DialogueLine with all fields."""
        from axiom_vox.multi_voice import DialogueLine

        line = DialogueLine(
            text="Welcome to the show!",
            voice_id="announcer",
            character_name="Host",
            emotion="enthusiastic",
            pause_before_ms=100,
            pause_after_ms=200,
        )
        assert line.text == "Welcome to the show!"
        assert line.voice_id == "announcer"
        assert line.character_name == "Host"
        assert line.emotion == "enthusiastic"
        assert line.pause_before_ms == 100
        assert line.pause_after_ms == 200

    def test_word_count(self):
        """Test word_count property."""
        from axiom_vox.multi_voice import DialogueLine

        line = DialogueLine(text="Hello world, how are you?", voice_id="test")
        assert line.word_count == 5

    def test_duration_estimate(self):
        """Test duration estimate calculation."""
        from axiom_vox.multi_voice import DialogueLine

        line = DialogueLine(
            text="One two three four five",
            voice_id="test",
            pause_before_ms=100,
            pause_after_ms=200,
        )
        # 5 words * 300ms/word + 100ms + 200ms = 1800ms
        assert line.duration_estimate_ms == 1800

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from axiom_vox.multi_voice import DialogueLine

        line = DialogueLine(
            text="Hello",
            voice_id="professional",
            emotion="calm",
        )
        d = line.to_dict()
        assert d["text"] == "Hello"
        assert d["voice_id"] == "professional"
        assert d["emotion"] == "calm"


class TestDialogueScript:
    """Tests for DialogueScript dataclass."""

    def test_basic_creation(self):
        """Test creating a DialogueScript."""
        from axiom_vox.multi_voice import DialogueScript, DialogueLine, TransitionStyle

        lines = [
            DialogueLine(text="Hello", voice_id="professional"),
            DialogueLine(text="World", voice_id="casual"),
        ]
        script = DialogueScript(lines=lines)

        assert len(script.lines) == 2
        assert script.default_transition == TransitionStyle.BREATH_PAUSE

    def test_voices_used(self):
        """Test voices_used property."""
        from axiom_vox.multi_voice import DialogueScript, DialogueLine

        lines = [
            DialogueLine(text="One", voice_id="professional"),
            DialogueLine(text="Two", voice_id="casual"),
            DialogueLine(text="Three", voice_id="professional"),
        ]
        script = DialogueScript(lines=lines)

        assert script.voices_used == {"professional", "casual"}

    def test_characters_used(self):
        """Test characters_used property."""
        from axiom_vox.multi_voice import DialogueScript, DialogueLine

        lines = [
            DialogueLine(text="One", voice_id="v1", character_name="Host"),
            DialogueLine(text="Two", voice_id="v2", character_name="Guest"),
            DialogueLine(text="Three", voice_id="v1"),  # No character
        ]
        script = DialogueScript(lines=lines)

        assert script.characters_used == {"Host", "Guest"}

    def test_total_duration(self):
        """Test total_duration_ms property."""
        from axiom_vox.multi_voice import DialogueScript, DialogueLine

        lines = [
            DialogueLine(text="One two", voice_id="v1"),  # 2 words * 300ms = 600ms
            DialogueLine(text="Three", voice_id="v2"),     # 1 word * 300ms = 300ms
        ]
        script = DialogueScript(lines=lines)

        assert script.total_duration_ms == 900

    def test_has_voice_switches(self):
        """Test has_voice_switches property."""
        from axiom_vox.multi_voice import DialogueScript, DialogueLine

        # No switches
        script1 = DialogueScript(lines=[
            DialogueLine(text="One", voice_id="v1"),
            DialogueLine(text="Two", voice_id="v1"),
        ])
        assert not script1.has_voice_switches

        # Has switches
        script2 = DialogueScript(lines=[
            DialogueLine(text="One", voice_id="v1"),
            DialogueLine(text="Two", voice_id="v2"),
        ])
        assert script2.has_voice_switches


class TestTransitionStyle:
    """Tests for TransitionStyle enum."""

    def test_all_styles_exist(self):
        """Test that all transition styles are defined."""
        from axiom_vox.multi_voice import TransitionStyle

        assert TransitionStyle.CROSSFADE.value == "crossfade"
        assert TransitionStyle.BREATH_PAUSE.value == "breath_pause"
        assert TransitionStyle.SILENCE.value == "silence"
        assert TransitionStyle.IMMEDIATE.value == "immediate"


class TestVoiceSwitch:
    """Tests for VoiceSwitch dataclass."""

    def test_creation(self):
        """Test creating a VoiceSwitch."""
        from axiom_vox.multi_voice import VoiceSwitch, VoiceSwitchType, TransitionStyle

        switch = VoiceSwitch(
            from_voice="professional",
            to_voice="casual",
            switch_type=VoiceSwitchType.DIALOGUE,
            transition_style=TransitionStyle.BREATH_PAUSE,
            word_index=5,
        )

        assert switch.from_voice == "professional"
        assert switch.to_voice == "casual"
        assert switch.switch_type == VoiceSwitchType.DIALOGUE
        assert switch.word_index == 5


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_merge_consecutive_lines(self):
        """Test merging consecutive lines with same voice."""
        from axiom_vox.multi_voice import DialogueLine, merge_consecutive_lines

        lines = [
            DialogueLine(text="Hello", voice_id="v1"),
            DialogueLine(text="there", voice_id="v1"),
            DialogueLine(text="world", voice_id="v2"),
        ]

        merged = merge_consecutive_lines(lines)

        assert len(merged) == 2
        assert merged[0].text == "Hello there"
        assert merged[0].voice_id == "v1"
        assert merged[1].text == "world"
        assert merged[1].voice_id == "v2"

    def test_parse_screenplay_format(self):
        """Test parsing screenplay format text."""
        from axiom_vox.multi_voice import parse_screenplay_format

        text = """
        HOST: Welcome to the show!
        GUEST: Thank you for having me.
        HOST: Let's begin.
        """

        script = parse_screenplay_format(text)

        assert len(script.lines) == 3
        assert script.lines[0].character_name == "HOST"
        assert script.lines[0].text == "Welcome to the show!"
        assert script.lines[1].character_name == "GUEST"

    def test_parse_chat_format(self):
        """Test parsing chat format text."""
        from axiom_vox.multi_voice import parse_chat_format

        text = """
        Alice: How are you?
        Bob: I'm doing well!
        Alice: Great to hear.
        """

        script = parse_chat_format(text)

        assert len(script.lines) == 3
        assert script.lines[0].character_name == "Alice"
        assert script.lines[1].character_name == "Bob"


# ============================================================================
# CHARACTER REGISTRY TESTS
# ============================================================================

class TestCharacterRegistry:
    """Tests for CharacterRegistry."""

    def test_register_character(self):
        """Test registering a character."""
        from axiom_vox.character_registry import CharacterRegistry

        registry = CharacterRegistry()
        mapping = registry.register(
            "Dr. Smith",
            voice_id="expert",
            default_emotion="confident",
        )

        assert mapping.character_name == "Dr. Smith"
        assert mapping.voice_id == "expert"
        assert mapping.default_emotion == "confident"

    def test_get_voice(self):
        """Test getting voice for character."""
        from axiom_vox.character_registry import CharacterRegistry

        registry = CharacterRegistry()
        registry.register("Host", voice_id="announcer")

        assert registry.get_voice("Host") == "announcer"
        assert registry.get_voice("host") == "announcer"  # Case insensitive
        assert registry.get_voice("Unknown") is None

    def test_get_or_assign(self):
        """Test get_or_assign functionality."""
        from axiom_vox.character_registry import CharacterRegistry

        registry = CharacterRegistry()
        registry.register("Known", voice_id="expert")

        # Known character
        assert registry.get_or_assign("Known") == "expert"

        # Unknown character - auto-assigned
        voice = registry.get_or_assign("Unknown")
        assert voice in registry.DEFAULT_VOICE_POOL

        # Same character returns same voice
        assert registry.get_or_assign("Unknown") == voice

    def test_auto_assign_voices(self):
        """Test auto-assigning voices to multiple characters."""
        from axiom_vox.character_registry import CharacterRegistry

        registry = CharacterRegistry()

        characters = ["Alice", "Bob", "Charlie"]
        assignments = registry.auto_assign_voices(characters)

        assert len(assignments) == 3
        assert all(v in registry.DEFAULT_VOICE_POOL for v in assignments.values())
        # All voices should be unique
        assert len(set(assignments.values())) == 3

    def test_list_characters(self):
        """Test listing characters."""
        from axiom_vox.character_registry import CharacterRegistry

        registry = CharacterRegistry()
        registry.register("Host", voice_id="announcer")
        registry.register("Guest", voice_id="casual")

        names = registry.list_character_names()
        assert "Host" in names
        assert "Guest" in names

    def test_unregister(self):
        """Test unregistering a character."""
        from axiom_vox.character_registry import CharacterRegistry

        registry = CharacterRegistry()
        registry.register("Host", voice_id="announcer")

        assert registry.unregister("Host") is True
        assert registry.get_voice("Host") is None
        assert registry.unregister("Unknown") is False


# ============================================================================
# SSML VOICE TAG TESTS
# ============================================================================

class TestSSMLVoiceTag:
    """Tests for SSML <voice> tag parsing."""

    def test_parse_voice_tag_with_axiom_voice(self):
        """Test parsing voice tag with axiom-voice attribute."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        ssml = '<speak><voice axiom-voice="professional">Hello world</voice></speak>'
        doc, warnings = parser.parse(ssml)

        assert len(doc.voice_spans) == 1
        assert doc.voice_spans[0].voice_id == "professional"
        assert doc.voice_spans[0].text == "Hello world"
        assert len(warnings) == 0

    def test_parse_voice_tag_with_name(self):
        """Test parsing voice tag with name attribute."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        ssml = '<speak><voice name="calm">Relaxed voice</voice></speak>'
        doc, warnings = parser.parse(ssml)

        assert len(doc.voice_spans) == 1
        assert doc.voice_spans[0].name == "calm"
        assert doc.voice_spans[0].text == "Relaxed voice"

    def test_parse_voice_tag_with_character(self):
        """Test parsing voice tag with character attribute."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        ssml = '<speak><voice character="Dr. Smith">Expert opinion</voice></speak>'
        doc, warnings = parser.parse(ssml)

        assert len(doc.voice_spans) == 1
        assert doc.voice_spans[0].character == "Dr. Smith"

    def test_parse_voice_tag_with_emotion(self):
        """Test parsing voice tag with emotion attribute."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        ssml = '<speak><voice axiom-voice="host" emotion="enthusiastic">Welcome!</voice></speak>'
        doc, warnings = parser.parse(ssml)

        assert doc.voice_spans[0].emotion == "enthusiastic"

    def test_parse_multiple_voice_tags(self):
        """Test parsing multiple voice tags."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        ssml = '''<speak>
            <voice axiom-voice="professional">First speaker.</voice>
            <voice axiom-voice="casual">Second speaker.</voice>
            <voice axiom-voice="expert">Third speaker.</voice>
        </speak>'''
        doc, warnings = parser.parse(ssml)

        assert len(doc.voice_spans) == 3
        assert doc.is_multi_voice
        assert set(doc.voices_used) == {"professional", "casual", "expert"}

    def test_ssml_voice_get_resolved_voice_id(self):
        """Test get_resolved_voice_id method."""
        from axiom_vox.ssml import SSMLVoice

        # Direct voice_id takes priority
        voice1 = SSMLVoice(voice_id="direct", name="named")
        assert voice1.get_resolved_voice_id() == "direct"

        # Name is mapped
        voice2 = SSMLVoice(name="professional")
        assert voice2.get_resolved_voice_id() == "professional"

        # Unknown name returns None
        voice3 = SSMLVoice(name="unknown_voice_xyz")
        assert voice3.get_resolved_voice_id() is None

        # Empty returns None
        voice4 = SSMLVoice()
        assert voice4.get_resolved_voice_id() is None


# ============================================================================
# TRANSITION PROCESSOR TESTS
# ============================================================================

class TestTransitionProcessor:
    """Tests for TransitionProcessor."""

    def test_generate_silence(self):
        """Test generating silence."""
        from axiom_vox.transition_processor import TransitionProcessor

        processor = TransitionProcessor(sample_rate=24000)
        silence = processor.generate_silence(duration_ms=100)

        expected_samples = 24000 * 100 // 1000  # 2400 samples
        expected_bytes = expected_samples * 2  # 16-bit audio

        assert len(silence) == expected_bytes
        assert silence == b'\x00' * expected_bytes

    def test_generate_breath_pause(self):
        """Test generating breath pause."""
        from axiom_vox.transition_processor import TransitionProcessor

        processor = TransitionProcessor(sample_rate=24000)
        breath = processor.generate_breath_pause(duration_ms=200)

        expected_samples = 24000 * 200 // 1000  # 4800 samples
        expected_bytes = expected_samples * 2

        assert len(breath) == expected_bytes
        # Breath pause has non-zero audio (ambient noise)
        assert breath != b'\x00' * expected_bytes

    def test_create_transition_immediate(self):
        """Test creating immediate transition."""
        from axiom_vox.transition_processor import TransitionProcessor, TransitionStyle

        processor = TransitionProcessor()
        result = processor.create_transition(TransitionStyle.IMMEDIATE)

        assert result.audio_bytes == b""
        assert result.duration_ms == 0
        assert result.is_empty

    def test_create_transition_breath_pause(self):
        """Test creating breath pause transition."""
        from axiom_vox.transition_processor import TransitionProcessor, TransitionStyle

        processor = TransitionProcessor()
        result = processor.create_transition(TransitionStyle.BREATH_PAUSE)

        assert len(result.audio_bytes) > 0
        assert result.duration_ms == 200
        assert not result.is_empty

    def test_apply_fade_in(self):
        """Test applying fade in."""
        from axiom_vox.transition_processor import TransitionProcessor
        import struct

        processor = TransitionProcessor(sample_rate=24000)

        # Create test audio (constant level)
        samples = [10000] * 240  # 10ms of audio
        audio = struct.pack(f"<{len(samples)}h", *samples)

        faded = processor.apply_fade_in(audio, fade_ms=5)

        # First sample should be lower than original
        faded_samples = struct.unpack(f"<{len(samples)}h", faded)
        assert abs(faded_samples[0]) < abs(samples[0])

    def test_apply_fade_out(self):
        """Test applying fade out."""
        from axiom_vox.transition_processor import TransitionProcessor
        import struct

        processor = TransitionProcessor(sample_rate=24000)

        # Create test audio (constant level)
        samples = [10000] * 240
        audio = struct.pack(f"<{len(samples)}h", *samples)

        faded = processor.apply_fade_out(audio, fade_ms=5)

        # Last sample should be lower than original
        faded_samples = struct.unpack(f"<{len(samples)}h", faded)
        assert abs(faded_samples[-1]) < abs(samples[-1])

    def test_get_duration_ms(self):
        """Test getting duration of audio."""
        from axiom_vox.transition_processor import TransitionProcessor

        processor = TransitionProcessor(sample_rate=24000)
        silence = processor.generate_silence(duration_ms=100)

        duration = processor.get_duration_ms(silence)
        assert abs(duration - 100) < 1  # Within 1ms


class TestTransitionConfig:
    """Tests for TransitionConfig."""

    def test_for_style(self):
        """Test getting config for style."""
        from axiom_vox.transition_processor import TransitionConfig, TransitionStyle

        breath_config = TransitionConfig.for_style(TransitionStyle.BREATH_PAUSE)
        assert breath_config.style == TransitionStyle.BREATH_PAUSE
        assert breath_config.duration_ms == 200

        crossfade_config = TransitionConfig.for_style(TransitionStyle.CROSSFADE)
        assert crossfade_config.style == TransitionStyle.CROSSFADE
        assert crossfade_config.crossfade_ms == 150


# ============================================================================
# MULTI-VOICE SYNTHESIZER TESTS
# ============================================================================

class TestMultiVoiceSynthesizer:
    """Tests for MultiVoiceSynthesizer."""

    def test_synthesize_script_basic(self):
        """Test basic script synthesis (placeholder mode)."""
        from axiom_vox.multi_voice import DialogueScript, DialogueLine
        from axiom_vox.multi_voice_synthesizer import MultiVoiceSynthesizer

        synthesizer = MultiVoiceSynthesizer()

        script = DialogueScript(lines=[
            DialogueLine(text="Hello", voice_id="professional"),
            DialogueLine(text="World", voice_id="casual"),
        ])

        result = synthesizer.synthesize_script(script)

        # Check result structure
        assert hasattr(result, 'success')
        assert hasattr(result, 'segments')
        assert hasattr(result, 'voice_switches')
        assert result.voices_used == ["professional", "casual"]

    def test_synthesizer_stats(self):
        """Test synthesizer statistics."""
        from axiom_vox.multi_voice_synthesizer import MultiVoiceSynthesizer

        synthesizer = MultiVoiceSynthesizer()
        stats = synthesizer.get_stats()

        assert "total_syntheses" in stats
        assert "total_voice_switches" in stats
        assert "cached_voice_configs" in stats

    def test_multi_voice_config(self):
        """Test MultiVoiceConfig defaults."""
        from axiom_vox.multi_voice_synthesizer import MultiVoiceConfig
        from axiom_vox.multi_voice import TransitionStyle

        config = MultiVoiceConfig()

        assert config.default_transition == TransitionStyle.BREATH_PAUSE
        assert config.breath_pause_ms == 200
        assert config.crossfade_ms == 150
        assert config.max_switches_per_minute == 30


class TestMultiVoiceSynthesizerSSML:
    """Tests for SSML-based multi-voice synthesis."""

    def test_synthesize_ssml_single_voice(self):
        """Test SSML synthesis with single voice."""
        from axiom_vox.multi_voice_synthesizer import MultiVoiceSynthesizer

        synthesizer = MultiVoiceSynthesizer()

        ssml = '<speak>Hello world</speak>'
        result = synthesizer.synthesize_ssml(ssml, default_voice_id="professional")

        assert len(result.segments) == 1
        assert not result.voice_switches  # No switches for single voice

    def test_synthesize_ssml_multi_voice(self):
        """Test SSML synthesis with multiple voices."""
        from axiom_vox.multi_voice_synthesizer import MultiVoiceSynthesizer

        synthesizer = MultiVoiceSynthesizer()

        ssml = '''<speak>
            <voice axiom-voice="professional">Welcome!</voice>
            <voice axiom-voice="casual">Thanks!</voice>
        </speak>'''

        result = synthesizer.synthesize_ssml(ssml)

        assert len(result.segments) == 2
        assert len(result.voice_switches) >= 1


# ============================================================================
# STREAMING MESSAGE TYPE TESTS
# ============================================================================

class TestStreamingMultiVoice:
    """Tests for multi-voice streaming support."""

    def test_message_type_voice_switch(self):
        """Test VOICE_SWITCH message type exists."""
        from axiom_vox.streaming import MessageType

        assert hasattr(MessageType, 'VOICE_SWITCH')
        assert MessageType.VOICE_SWITCH.value == "voice_switch"

    def test_audio_chunk_voice_id(self):
        """Test AudioChunk has voice_id field."""
        from axiom_vox.streaming import AudioChunk

        chunk = AudioChunk(
            data=b"test",
            index=0,
            timestamp_ms=0,
            duration_ms=100,
            voice_id="professional",
            is_voice_transition=False,
        )

        assert chunk.voice_id == "professional"
        assert not chunk.is_voice_transition

    def test_audio_chunk_to_dict_includes_voice(self):
        """Test AudioChunk.to_dict includes voice fields."""
        from axiom_vox.streaming import AudioChunk

        chunk = AudioChunk(
            data=b"test",
            index=0,
            timestamp_ms=0,
            duration_ms=100,
            voice_id="casual",
            is_voice_transition=True,
        )

        d = chunk.to_dict()
        assert d["voice_id"] == "casual"
        assert d["is_voice_transition"] is True

    def test_stream_message_voice_switch(self):
        """Test StreamMessage with voice_switch payload."""
        from axiom_vox.streaming import StreamMessage, MessageType

        msg = StreamMessage(
            type=MessageType.VOICE_SWITCH,
            request_id="test-123",
            voice_switch={
                "from_voice": "professional",
                "to_voice": "casual",
                "transition_style": "breath_pause",
            }
        )

        json_data = msg.to_json()
        assert json_data["type"] == "voice_switch"
        assert "voice_switch" in json_data
        assert json_data["voice_switch"]["from_voice"] == "professional"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMultiVoiceIntegration:
    """Integration tests for multi-voice synthesis."""

    def test_full_pipeline_dialogue(self):
        """Test full dialogue synthesis pipeline."""
        from axiom_vox.multi_voice import DialogueScript, DialogueLine, TransitionStyle
        from axiom_vox.character_registry import CharacterRegistry
        from axiom_vox.multi_voice_synthesizer import MultiVoiceSynthesizer

        # Setup character registry
        registry = CharacterRegistry()
        registry.register("Host", voice_id="announcer")
        registry.register("Guest", voice_id="casual", default_emotion="friendly")

        # Create synthesizer with registry
        synthesizer = MultiVoiceSynthesizer(character_registry=registry)

        # Create script
        script = DialogueScript(
            lines=[
                DialogueLine(text="Welcome to the show!", character_name="Host"),
                DialogueLine(text="Thanks for having me!", character_name="Guest"),
            ],
            default_transition=TransitionStyle.BREATH_PAUSE,
        )

        # Synthesize
        result = synthesizer.synthesize_script(script)

        # Verify
        assert len(result.segments) == 2
        assert "announcer" in result.voices_used
        assert "casual" in result.voices_used

    def test_screenplay_to_synthesis(self):
        """Test screenplay format to synthesis."""
        from axiom_vox.multi_voice import parse_screenplay_format
        from axiom_vox.multi_voice_synthesizer import MultiVoiceSynthesizer

        screenplay = """
        NARRATOR: Once upon a time.
        HERO: I will save the day!
        VILLAIN: Not if I can help it.
        """

        script = parse_screenplay_format(screenplay)
        synthesizer = MultiVoiceSynthesizer()
        result = synthesizer.synthesize_script(script)

        assert len(result.segments) == 3

    def test_module_exports(self):
        """Test that multi-voice modules are properly exported."""
        import axiom_vox

        # Data models
        assert hasattr(axiom_vox, 'DialogueLine')
        assert hasattr(axiom_vox, 'DialogueScript')
        assert hasattr(axiom_vox, 'TransitionStyle')
        assert hasattr(axiom_vox, 'VoiceSwitch')

        # Registry
        assert hasattr(axiom_vox, 'CharacterRegistry')
        assert hasattr(axiom_vox, 'get_character_registry')

        # Transitions
        assert hasattr(axiom_vox, 'TransitionProcessor')
        assert hasattr(axiom_vox, 'generate_breath_pause')

        # Synthesizer
        assert hasattr(axiom_vox, 'MultiVoiceSynthesizer')
        assert hasattr(axiom_vox, 'get_multi_voice_synthesizer')

        # Version
        assert axiom_vox.__version__ == "0.9.0"


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_script(self):
        """Test handling empty script."""
        from axiom_vox.multi_voice import DialogueScript

        script = DialogueScript(lines=[])

        assert script.voices_used == set()
        assert script.total_duration_ms == 0

    def test_single_line_script(self):
        """Test script with single line."""
        from axiom_vox.multi_voice import DialogueScript, DialogueLine

        script = DialogueScript(lines=[
            DialogueLine(text="Solo", voice_id="professional"),
        ])

        assert not script.has_voice_switches
        assert script.voices_used == {"professional"}

    def test_empty_text_line(self):
        """Test line with empty text."""
        from axiom_vox.multi_voice import DialogueLine

        line = DialogueLine(text="", voice_id="professional")
        assert line.word_count == 0
        assert line.duration_estimate_ms == 0

    def test_registry_case_insensitivity(self):
        """Test character registry is case insensitive."""
        from axiom_vox.character_registry import CharacterRegistry

        registry = CharacterRegistry()
        registry.register("Dr. SMITH", voice_id="expert")

        assert registry.get_voice("dr. smith") == "expert"
        assert registry.get_voice("DR. SMITH") == "expert"
        assert registry.get_voice("Dr. Smith") == "expert"

    def test_ssml_empty_voice_tag(self):
        """Test parsing empty voice tag."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        ssml = '<speak><voice axiom-voice="test"></voice></speak>'
        doc, _ = parser.parse(ssml)

        # Empty voice tag should not create span
        assert len(doc.voice_spans) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
