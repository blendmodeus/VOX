"""
Tests for SSML Parser and Generator
------------------------------------

Tests for:
- SSMLParser: Parse SSML markup into SSMLDocument
- SSMLGenerator: Generate SSML from ProsodyTarget
- SSMLDocument: Data structure integrity
- Edge cases and error handling
"""

import pytest
from typing import List


# ============================================================================
# SSML PARSER TESTS
# ============================================================================

class TestSSMLParser:
    """Tests for SSMLParser."""

    def test_basic_speak_tag(self):
        """Test parsing basic <speak> wrapper."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        doc, warnings = parser.parse('<speak>Hello world</speak>')

        assert doc.plain_text == 'Hello world'
        assert doc.word_list == ['Hello', 'world']
        assert len(warnings) == 0

    def test_plain_text_fallback(self):
        """Test parsing plain text without SSML."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        doc, warnings = parser.parse('Just plain text')

        assert doc.plain_text == 'Just plain text'
        assert len(doc.word_list) == 3

    def test_break_element_with_time(self):
        """Test parsing <break time="500ms"/>."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        doc, _ = parser.parse('<speak>Hello<break time="500ms"/> world</speak>')

        assert 'Hello' in doc.plain_text
        assert 'world' in doc.plain_text
        assert len(doc.breaks) == 1
        assert doc.breaks[0].time_ms == 500
        assert 0 in doc.pause_locations
        assert doc.pause_locations[0] == 0.5  # 500ms = 0.5s

    def test_break_element_with_strength(self):
        """Test parsing <break strength="strong"/>."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        doc, _ = parser.parse('<speak>Hello<break strength="strong"/> world</speak>')

        assert len(doc.breaks) == 1
        assert doc.breaks[0].strength == 'strong'
        assert doc.breaks[0].time_ms == 750  # strong = 750ms

    def test_emphasis_element(self):
        """Test parsing <emphasis>."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        doc, _ = parser.parse('<speak>This is <emphasis level="strong">very important</emphasis> text.</speak>')

        assert len(doc.emphases) == 1
        assert doc.emphases[0].text == 'very important'
        assert doc.emphases[0].level == 'strong'
        assert len(doc.emphasis_words) == 2  # 'very' and 'important'

    def test_emphasis_levels(self):
        """Test different emphasis levels."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()

        for level in ['strong', 'moderate', 'reduced', 'none']:
            doc, _ = parser.parse(f'<speak><emphasis level="{level}">word</emphasis></speak>')
            assert doc.emphases[0].level == level

    def test_prosody_element(self):
        """Test parsing <prosody>."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        doc, _ = parser.parse('<speak><prosody rate="slow" pitch="+2st">Slow and high</prosody></speak>')

        # Prosody is captured and text is extracted
        assert 'Slow' in doc.plain_text
        assert 'high' in doc.plain_text
        assert len(doc.word_list) == 3

        # Prosody spans are created for text within prosody tags
        assert len(doc.prosody_spans) >= 1
        prosody = doc.prosody_spans[0]
        assert prosody.rate == 'slow'
        assert prosody.pitch == '+2st'

    def test_prosody_rate_parsing(self):
        """Test rate value parsing."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()

        # Keyword rates
        assert parser.parse_rate('x-slow') == 0.5
        assert parser.parse_rate('slow') == 0.75
        assert parser.parse_rate('medium') == 1.0
        assert parser.parse_rate('fast') == 1.25
        assert parser.parse_rate('x-fast') == 1.5

        # Percentage rates
        assert parser.parse_rate('120%') == 1.2
        assert parser.parse_rate('80%') == 0.8

    def test_prosody_pitch_parsing(self):
        """Test pitch value parsing."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()

        # Keyword pitches
        assert parser.parse_pitch('x-low') == -4
        assert parser.parse_pitch('low') == -2
        assert parser.parse_pitch('high') == 2
        assert parser.parse_pitch('x-high') == 4

        # Semitone values
        assert parser.parse_pitch('+2st') == 2.0
        assert parser.parse_pitch('-1.5st') == -1.5

    def test_say_as_element(self):
        """Test parsing <say-as>."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        doc, _ = parser.parse('<speak>Call <say-as interpret-as="telephone">+1-800-555-1234</say-as></speak>')

        assert len(doc.say_as_spans) == 1
        assert doc.say_as_spans[0].interpret_as == 'telephone'
        assert '+1-800-555-1234' in doc.say_as_spans[0].text

    def test_sub_element(self):
        """Test parsing <sub> for pronunciation substitution."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        doc, _ = parser.parse('<speak>The <sub alias="World Wide Web Consortium">W3C</sub> defines standards.</speak>')

        assert len(doc.substitutions) == 1
        assert doc.substitutions[0].original == 'W3C'
        assert doc.substitutions[0].alias == 'World Wide Web Consortium'
        # Plain text should contain the alias
        assert 'World Wide Web Consortium' in doc.plain_text

    def test_empty_input(self):
        """Test parsing empty input."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        doc, _ = parser.parse('')

        assert doc.plain_text == ''
        assert len(doc.word_list) == 0

    def test_malformed_xml_fallback(self):
        """Test fallback for malformed XML."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        doc, warnings = parser.parse('<speak>Unclosed tag <break>')

        assert len(warnings) > 0
        assert 'Hello' in doc.plain_text or 'Unclosed' in doc.plain_text

    def test_nested_elements(self):
        """Test parsing nested SSML elements."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        ssml = '''<speak>
            <prosody rate="slow">
                This is <emphasis level="strong">important</emphasis> text.
            </prosody>
        </speak>'''
        doc, _ = parser.parse(ssml)

        assert 'important' in doc.plain_text
        assert len(doc.prosody_spans) >= 1
        assert len(doc.emphases) == 1


# ============================================================================
# SSML GENERATOR TESTS
# ============================================================================

class TestSSMLGenerator:
    """Tests for SSMLGenerator."""

    def test_generate_simple(self):
        """Test simple SSML generation."""
        from axiom_vox.ssml import SSMLGenerator

        generator = SSMLGenerator()
        ssml = generator.generate_simple(
            text="Hello world",
            rate=1.0,
        )

        assert '<speak>' in ssml
        assert '</speak>' in ssml
        assert 'Hello' in ssml
        assert 'world' in ssml

    def test_generate_with_rate(self):
        """Test SSML generation with speaking rate."""
        from axiom_vox.ssml import SSMLGenerator

        generator = SSMLGenerator()
        ssml = generator.generate_simple(
            text="Fast text",
            rate=1.5,
        )

        assert 'rate="150%"' in ssml

    def test_generate_with_pitch(self):
        """Test SSML generation with pitch."""
        from axiom_vox.ssml import SSMLGenerator

        generator = SSMLGenerator()
        ssml = generator.generate_simple(
            text="High text",
            pitch=2.0,
        )

        assert 'pitch="+2st"' in ssml

    def test_generate_with_pauses(self):
        """Test SSML generation with pause locations."""
        from axiom_vox.ssml import SSMLGenerator

        generator = SSMLGenerator()
        ssml = generator.generate_simple(
            text="Hello beautiful world today",
            pauses={1: 0.5},  # 500ms after word 1
        )

        assert '<break time="500ms"/>' in ssml

    def test_generate_with_emphases(self):
        """Test SSML generation with emphasis."""
        from axiom_vox.ssml import SSMLGenerator

        generator = SSMLGenerator()
        ssml = generator.generate_simple(
            text="This is important text",
            emphases=[2],  # word index 2 = "important"
        )

        assert '<emphasis level="moderate">important</emphasis>' in ssml

    def test_generate_from_prosody_target(self):
        """Test SSML generation from ProsodyTarget."""
        from axiom_vox.ssml import SSMLGenerator
        from axiom_vox.prosody_director import ProsodyTarget

        generator = SSMLGenerator()
        target = ProsodyTarget(
            speaking_rate=0.9,
            pitch_base=1.5,
            pause_locations={1: 0.3},
            emphasis_words=[0],
        )

        ssml = generator.generate("Hello world today", target)

        assert '<speak>' in ssml
        assert 'rate="90%"' in ssml
        assert 'pitch="+1.5st"' in ssml

    def test_xml_escaping(self):
        """Test that special characters are escaped."""
        from axiom_vox.ssml import SSMLGenerator

        generator = SSMLGenerator()
        ssml = generator.generate_simple(
            text="5 < 10 & 10 > 5",
        )

        assert '&lt;' in ssml
        assert '&gt;' in ssml
        assert '&amp;' in ssml


# ============================================================================
# SSML DOCUMENT TESTS
# ============================================================================

class TestSSMLDocument:
    """Tests for SSMLDocument dataclass."""

    def test_document_creation(self):
        """Test creating an SSMLDocument."""
        from axiom_vox.ssml import SSMLDocument

        doc = SSMLDocument(
            plain_text="Hello world",
            word_list=["Hello", "world"],
            pause_locations={0: 0.5},
            emphasis_words=[1],
        )

        assert doc.plain_text == "Hello world"
        assert len(doc.word_list) == 2
        assert doc.pause_locations[0] == 0.5
        assert 1 in doc.emphasis_words

    def test_document_to_dict(self):
        """Test document serialization."""
        from axiom_vox.ssml import SSMLDocument

        doc = SSMLDocument(
            plain_text="Test text",
            word_list=["Test", "text"],
        )

        d = doc.to_dict()

        assert d["plain_text"] == "Test text"
        assert d["word_count"] == 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSSMLIntegration:
    """Integration tests for SSML workflow."""

    def test_parse_and_regenerate(self):
        """Test parsing SSML and regenerating it."""
        from axiom_vox.ssml import SSMLParser, SSMLGenerator

        original_ssml = '<speak>Hello <emphasis level="strong">world</emphasis>!</speak>'

        parser = SSMLParser()
        doc, _ = parser.parse(original_ssml)

        generator = SSMLGenerator()
        regenerated = generator.generate_simple(
            text=doc.plain_text,
            emphases=doc.emphasis_words,
        )

        assert '<emphasis' in regenerated
        assert 'world' in regenerated

    def test_ssml_round_trip(self):
        """Test that parsed SSML preserves semantic information."""
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()

        ssml = '''<speak>
            <prosody rate="slow">
                Please <emphasis level="strong">listen carefully</emphasis>.
                <break time="500ms"/>
                This is important.
            </prosody>
        </speak>'''

        doc, _ = parser.parse(ssml)

        # Check semantic preservation
        assert 'listen' in doc.plain_text
        assert 'carefully' in doc.plain_text
        assert 'important' in doc.plain_text
        assert len(doc.breaks) == 1
        assert len(doc.emphases) == 1


# ============================================================================
# MODULE EXPORTS TEST
# ============================================================================

class TestSSMLExports:
    """Tests for module exports."""

    def test_ssml_exports(self):
        """Test that all public classes are exported."""
        from axiom_vox.ssml import (
            SSMLParser,
            SSMLGenerator,
            SSMLDocument,
            SSMLBreak,
            SSMLEmphasis,
            SSMLProsody,
            SSMLSayAs,
            SSMLSub,
        )

        assert SSMLParser is not None
        assert SSMLGenerator is not None
        assert SSMLDocument is not None
        assert SSMLBreak is not None
        assert SSMLEmphasis is not None


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
