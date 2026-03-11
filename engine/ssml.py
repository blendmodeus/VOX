"""
AXIOM VOX SSML Support
----------------------

W3C SSML 1.1 subset parser and generator for expressive TTS.

Supported elements:
- <speak> - Document wrapper
- <break time="500ms"/> - Pauses
- <emphasis level="strong|moderate|reduced"> - Word emphasis
- <prosody rate="slow" pitch="+2st" volume="loud"> - Prosody control
- <say-as interpret-as="date|time|telephone|currency"> - Interpretation hints
- <sub alias="pronunciation"> - Pronunciation substitutions

Usage:
    from axiom_vox.ssml import SSMLParser, SSMLGenerator

    # Parse SSML input
    parser = SSMLParser()
    doc, warnings = parser.parse('<speak>Hello <break time="500ms"/> world!</speak>')

    # Generate SSML from ProsodyTarget
    generator = SSMLGenerator()
    ssml = generator.generate("Hello world", prosody_target)
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from html import unescape

if TYPE_CHECKING:
    from axiom_vox.prosody_director import ProsodyTarget
    from axiom_vox.prosody_guardrails import EmotionalIntent


# ============================================================================
# SSML DATA STRUCTURES
# ============================================================================

@dataclass
class SSMLBreak:
    """Represents a <break> element."""
    time_ms: int  # Duration in milliseconds
    strength: Optional[str] = None  # "none", "x-weak", "weak", "medium", "strong", "x-strong"
    word_index: int = 0  # Position in text (after which word)


@dataclass
class SSMLEmphasis:
    """Represents an <emphasis> element."""
    text: str
    level: str  # "strong", "moderate", "reduced", "none"
    start_word_index: int = 0
    end_word_index: int = 0


@dataclass
class SSMLProsody:
    """Represents a <prosody> element."""
    text: str = ""
    rate: Optional[str] = None      # "x-slow", "slow", "medium", "fast", "x-fast", or percentage
    pitch: Optional[str] = None     # "+Xst", "-Xst", "x-low" to "x-high"
    volume: Optional[str] = None    # "silent", "x-soft", "soft", "medium", "loud", "x-loud", or dB
    range: Optional[str] = None     # Pitch range
    duration: Optional[str] = None  # Target duration
    contour: Optional[str] = None   # Pitch contour
    start_word_index: int = 0
    end_word_index: int = 0


@dataclass
class SSMLSayAs:
    """Represents a <say-as> element."""
    text: str
    interpret_as: str  # "date", "time", "telephone", "currency", "cardinal", "ordinal", "characters"
    format: Optional[str] = None  # e.g., "mdy" for dates
    detail: Optional[str] = None
    start_word_index: int = 0
    end_word_index: int = 0


@dataclass
class SSMLSub:
    """Represents a <sub> element for pronunciation substitution."""
    original: str
    alias: str  # The pronunciation replacement
    word_index: int = 0


@dataclass
class SSMLVoice:
    """
    Represents a <voice> element for multi-voice synthesis.

    Supports W3C SSML voice attributes plus AXIOM extensions.

    Attributes:
        name: W3C voice name (e.g., "en-US-Standard-A")
        voice_id: AXIOM voice ID (axiom-voice attribute)
        character: Character name for registry lookup
        gender: "male", "female", "neutral"
        age: "child", "adult", "elder"
        emotion: Emotion preset for this voice segment
        text: Text content within this voice element
        start_word_index: Starting position in word list
        end_word_index: Ending position in word list

    v0.9.0: Multi-voice synthesis support
    """
    # W3C SSML attributes
    name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    variant: Optional[str] = None
    languages: List[str] = field(default_factory=list)

    # AXIOM extensions
    voice_id: Optional[str] = None  # axiom-voice attribute
    character: Optional[str] = None  # character registry lookup
    emotion: Optional[str] = None  # emotion preset

    # Content
    text: str = ""
    start_word_index: int = 0
    end_word_index: int = 0

    def get_resolved_voice_id(self) -> Optional[str]:
        """
        Get the resolved voice ID.

        Priority: voice_id > name > None
        """
        if self.voice_id:
            return self.voice_id
        if self.name:
            # Map W3C names to AXIOM voices
            name_lower = self.name.lower()
            name_map = {
                "professional": "professional",
                "conversational": "conversational",
                "expert": "expert",
                "calm": "calm",
                "announcer": "announcer",
                "guide": "guide",
                "casual": "casual",
                "corporate": "corporate",
            }
            return name_map.get(name_lower)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "voice_id": self.voice_id,
            "character": self.character,
            "gender": self.gender,
            "age": self.age,
            "emotion": self.emotion,
            "text": self.text,
            "start_word_index": self.start_word_index,
            "end_word_index": self.end_word_index,
        }


@dataclass
class SSMLDocument:
    """
    Parsed SSML document representation.

    Contains the extracted plain text and all SSML annotations
    with word-level positioning for prosody application.

    v0.9.0: Added voice_spans for multi-voice synthesis.
    """
    plain_text: str  # Extracted text without markup
    word_list: List[str] = field(default_factory=list)  # Tokenized words
    breaks: List[SSMLBreak] = field(default_factory=list)
    emphases: List[SSMLEmphasis] = field(default_factory=list)
    prosody_spans: List[SSMLProsody] = field(default_factory=list)
    say_as_spans: List[SSMLSayAs] = field(default_factory=list)
    substitutions: List[SSMLSub] = field(default_factory=list)
    voice_spans: List[SSMLVoice] = field(default_factory=list)  # v0.9.0: Multi-voice

    # Derived representations (for integration with ProsodyTarget)
    pause_locations: Dict[int, float] = field(default_factory=dict)  # word_idx -> pause_seconds
    emphasis_words: List[int] = field(default_factory=list)  # word indices

    @property
    def is_multi_voice(self) -> bool:
        """Check if document contains multiple voices."""
        return len(self.voice_spans) > 1

    @property
    def voices_used(self) -> List[str]:
        """Get list of voice IDs used in document."""
        return [v.get_resolved_voice_id() for v in self.voice_spans if v.get_resolved_voice_id()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for debugging/logging."""
        return {
            "plain_text": self.plain_text,
            "word_count": len(self.word_list),
            "breaks": len(self.breaks),
            "emphases": len(self.emphases),
            "prosody_spans": len(self.prosody_spans),
            "voice_spans": len(self.voice_spans),
            "is_multi_voice": self.is_multi_voice,
            "voices_used": self.voices_used,
            "pause_locations": self.pause_locations,
            "emphasis_words": self.emphasis_words,
        }


# ============================================================================
# SSML PARSER
# ============================================================================

class SSMLParser:
    """
    Parses W3C SSML 1.1 subset into AXIOM VOX internal representation.

    Supported elements are parsed and converted to SSMLDocument.
    Unsupported elements are stripped but their text content is preserved.
    """

    # Time pattern: "500ms", "0.5s", "500"
    TIME_PATTERN = re.compile(r'^(\d+(?:\.\d+)?)(ms|s)?$')

    # Rate values to multipliers
    RATE_MAP = {
        "x-slow": 0.5,
        "slow": 0.75,
        "medium": 1.0,
        "fast": 1.25,
        "x-fast": 1.5,
    }

    # Pitch values to semitones
    PITCH_MAP = {
        "x-low": -4,
        "low": -2,
        "medium": 0,
        "high": 2,
        "x-high": 4,
    }

    # Volume values (relative)
    VOLUME_MAP = {
        "silent": 0.0,
        "x-soft": 0.25,
        "soft": 0.5,
        "medium": 1.0,
        "loud": 1.5,
        "x-loud": 2.0,
    }

    # Emphasis level to weight
    EMPHASIS_MAP = {
        "strong": 1.0,
        "moderate": 0.6,
        "reduced": 0.3,
        "none": 0.0,
    }

    # Break strength to milliseconds
    STRENGTH_MAP = {
        "none": 0,
        "x-weak": 100,
        "weak": 200,
        "medium": 400,
        "strong": 750,
        "x-strong": 1000,
    }

    def parse(self, ssml: str) -> Tuple[SSMLDocument, List[str]]:
        """
        Parse SSML string into SSMLDocument.

        Args:
            ssml: SSML markup string (with or without <speak> wrapper)

        Returns:
            Tuple of (SSMLDocument, list of warnings)
        """
        warnings = []

        # Handle empty input
        if not ssml or not ssml.strip():
            return SSMLDocument(plain_text="", word_list=[]), []

        ssml = ssml.strip()

        # Wrap plain text in <speak> tags if needed
        if not ssml.startswith('<speak') and not ssml.startswith('<?xml'):
            # Check if it contains any XML-like content
            if '<' in ssml:
                ssml = f'<speak>{ssml}</speak>'
            else:
                # Plain text, no SSML
                words = ssml.split()
                return SSMLDocument(
                    plain_text=ssml,
                    word_list=words,
                ), []

        # Parse XML
        try:
            root = ET.fromstring(ssml)
        except ET.ParseError as e:
            warnings.append(f"SSML parse error: {e}")
            # Fallback: strip tags and return plain text
            plain_text = re.sub(r'<[^>]+>', ' ', ssml)
            plain_text = ' '.join(plain_text.split())
            return SSMLDocument(
                plain_text=plain_text,
                word_list=plain_text.split(),
            ), warnings

        # Extract all elements
        breaks: List[SSMLBreak] = []
        emphases: List[SSMLEmphasis] = []
        prosody_spans: List[SSMLProsody] = []
        say_as_spans: List[SSMLSayAs] = []
        substitutions: List[SSMLSub] = []
        voice_spans: List[SSMLVoice] = []

        # Track word position as we walk the tree
        word_position = [0]  # Mutable container for closure
        plain_text_parts: List[str] = []

        def get_tag_name(elem) -> str:
            """Get tag name, stripping namespace if present."""
            tag = elem.tag if isinstance(elem.tag, str) else ''
            # Handle namespace: {http://...}tagname
            if tag.startswith('{'):
                tag = tag.split('}')[-1]
            return tag.lower()

        def walk(elem, current_prosody: Optional[Dict[str, str]] = None):
            """Recursively walk SSML tree."""
            tag = get_tag_name(elem)

            # Elements that handle their own content (don't process elem.text separately)
            self_handling_tags = {'prosody', 'emphasis', 'say-as', 'sub', 'voice'}

            # Handle text before children (but not for self-handling elements)
            if elem.text and tag not in self_handling_tags:
                text = elem.text
                # Normalize whitespace but preserve some spacing
                words = text.split()
                if words:
                    start_idx = word_position[0]

                    # Apply inherited prosody to these words
                    if current_prosody and any(current_prosody.values()):
                        prosody_spans.append(SSMLProsody(
                            text=' '.join(words),
                            rate=current_prosody.get('rate'),
                            pitch=current_prosody.get('pitch'),
                            volume=current_prosody.get('volume'),
                            start_word_index=start_idx,
                            end_word_index=start_idx + len(words) - 1,
                        ))

                    plain_text_parts.extend(words)
                    word_position[0] += len(words)

            # Process element-specific logic
            if tag == 'break':
                time_ms = self._parse_time(
                    elem.get('time', ''),
                    elem.get('strength', 'medium')
                )
                breaks.append(SSMLBreak(
                    time_ms=time_ms,
                    strength=elem.get('strength'),
                    word_index=max(0, word_position[0] - 1),
                ))

            elif tag == 'emphasis':
                level = elem.get('level', 'moderate')
                inner_text = self._get_inner_text(elem)
                words = inner_text.split()
                if words:
                    start_idx = word_position[0]
                    emphases.append(SSMLEmphasis(
                        text=inner_text,
                        level=level,
                        start_word_index=start_idx,
                        end_word_index=start_idx + len(words) - 1,
                    ))
                    plain_text_parts.extend(words)
                    word_position[0] += len(words)
                return  # Don't recurse - we handled the content

            elif tag == 'prosody':
                # Extract prosody attributes and pass to children
                new_prosody = {
                    'rate': elem.get('rate') or (current_prosody.get('rate') if current_prosody else None),
                    'pitch': elem.get('pitch') or (current_prosody.get('pitch') if current_prosody else None),
                    'volume': elem.get('volume') or (current_prosody.get('volume') if current_prosody else None),
                }

                # Track start position for prosody span
                start_idx = word_position[0]

                # Process direct text content of prosody element
                if elem.text:
                    text_words = elem.text.split()
                    if text_words:
                        plain_text_parts.extend(text_words)
                        word_position[0] += len(text_words)

                # Recursively process children with prosody context
                for child in elem:
                    walk(child, new_prosody)
                    # Handle tail text after child elements
                    if child.tail:
                        tail_words = child.tail.split()
                        if tail_words:
                            plain_text_parts.extend(tail_words)
                            word_position[0] += len(tail_words)

                # Create prosody span for the entire content
                end_idx = word_position[0] - 1
                if end_idx >= start_idx:
                    inner_text = ' '.join(plain_text_parts[start_idx:end_idx + 1])
                    prosody_spans.append(SSMLProsody(
                        text=inner_text,
                        rate=new_prosody.get('rate'),
                        pitch=new_prosody.get('pitch'),
                        volume=new_prosody.get('volume'),
                        start_word_index=start_idx,
                        end_word_index=end_idx,
                    ))
                return  # Already processed children

            elif tag == 'say-as':
                inner_text = self._get_inner_text(elem)
                words = inner_text.split()
                if words:
                    start_idx = word_position[0]
                    say_as_spans.append(SSMLSayAs(
                        text=inner_text,
                        interpret_as=elem.get('interpret-as', 'characters'),
                        format=elem.get('format'),
                        detail=elem.get('detail'),
                        start_word_index=start_idx,
                        end_word_index=start_idx + len(words) - 1,
                    ))
                    plain_text_parts.extend(words)
                    word_position[0] += len(words)
                return  # Don't recurse

            elif tag == 'sub':
                original = self._get_inner_text(elem)
                alias = elem.get('alias', original)
                substitutions.append(SSMLSub(
                    original=original,
                    alias=alias,
                    word_index=word_position[0],
                ))
                # Use alias in plain text
                alias_words = alias.split()
                plain_text_parts.extend(alias_words)
                word_position[0] += len(alias_words)
                return  # Don't recurse into <sub>

            elif tag == 'voice':
                # v0.9.0: Multi-voice support
                # Extract voice attributes
                voice_name = elem.get('name')
                voice_id = elem.get('axiom-voice') or elem.get('voice-id')
                gender = elem.get('gender')
                age = elem.get('age')
                character = elem.get('character')
                emotion = elem.get('emotion')
                variant = elem.get('variant')
                lang = elem.get('xml:lang') or elem.get('lang')

                # Track start position
                start_idx = word_position[0]

                # Process direct text content
                if elem.text:
                    text_words = elem.text.split()
                    if text_words:
                        plain_text_parts.extend(text_words)
                        word_position[0] += len(text_words)

                # Recursively process children (voice can contain other elements)
                for child in elem:
                    walk(child, current_prosody)
                    if child.tail:
                        tail_words = child.tail.split()
                        if tail_words:
                            plain_text_parts.extend(tail_words)
                            word_position[0] += len(tail_words)

                # Create voice span for this content
                end_idx = word_position[0] - 1
                if end_idx >= start_idx:
                    inner_text = ' '.join(plain_text_parts[start_idx:end_idx + 1])
                    voice_spans.append(SSMLVoice(
                        name=voice_name,
                        voice_id=voice_id,
                        character=character,
                        gender=gender,
                        age=age,
                        variant=variant,
                        emotion=emotion,
                        languages=[lang] if lang else [],
                        text=inner_text,
                        start_word_index=start_idx,
                        end_word_index=end_idx,
                    ))
                return  # Already processed children

            # Recurse into children
            for child in elem:
                walk(child, current_prosody)
                # Handle tail text (text after child element)
                if child.tail:
                    tail_words = child.tail.split()
                    if tail_words:
                        plain_text_parts.extend(tail_words)
                        word_position[0] += len(tail_words)

        walk(root)

        # Build final document
        plain_text = ' '.join(plain_text_parts)
        word_list = plain_text_parts

        # Convert breaks to pause_locations
        pause_locations: Dict[int, float] = {}
        for brk in breaks:
            idx = brk.word_index
            seconds = brk.time_ms / 1000.0
            # Accumulate pauses at same position
            pause_locations[idx] = pause_locations.get(idx, 0) + seconds

        # Convert emphases to word indices
        emphasis_words: List[int] = []
        for emp in emphases:
            for i in range(emp.start_word_index, min(emp.end_word_index + 1, len(word_list))):
                if i not in emphasis_words:
                    emphasis_words.append(i)

        return SSMLDocument(
            plain_text=plain_text,
            word_list=word_list,
            breaks=breaks,
            emphases=emphases,
            prosody_spans=prosody_spans,
            say_as_spans=say_as_spans,
            substitutions=substitutions,
            voice_spans=voice_spans,
            pause_locations=pause_locations,
            emphasis_words=sorted(emphasis_words),
        ), warnings

    def _parse_time(self, time_str: str, strength: str = "medium") -> int:
        """Parse time string to milliseconds."""
        if time_str:
            match = self.TIME_PATTERN.match(time_str)
            if match:
                value = float(match.group(1))
                unit = match.group(2) or 'ms'
                if unit == 's':
                    return int(value * 1000)
                return int(value)

        # Fallback to strength
        return self.STRENGTH_MAP.get(strength, 400)

    def _get_inner_text(self, elem) -> str:
        """Get all text content from element and children."""
        return ''.join(elem.itertext())

    def parse_rate(self, rate_str: str) -> float:
        """Parse rate string to multiplier."""
        if not rate_str:
            return 1.0
        if rate_str in self.RATE_MAP:
            return self.RATE_MAP[rate_str]
        # Handle percentage
        if rate_str.endswith('%'):
            try:
                return float(rate_str[:-1]) / 100
            except ValueError:
                return 1.0
        return 1.0

    def parse_pitch(self, pitch_str: str) -> float:
        """Parse pitch string to semitones."""
        if not pitch_str:
            return 0.0
        if pitch_str in self.PITCH_MAP:
            return float(self.PITCH_MAP[pitch_str])
        # Handle semitones: "+2st", "-1.5st"
        if pitch_str.endswith('st'):
            try:
                return float(pitch_str[:-2])
            except ValueError:
                return 0.0
        # Handle Hz offset (not fully supported)
        if pitch_str.endswith('Hz'):
            try:
                # Rough approximation: 100Hz ~ 1 semitone
                return float(pitch_str[:-2]) / 100
            except ValueError:
                return 0.0
        return 0.0


# ============================================================================
# SSML GENERATOR
# ============================================================================

class SSMLGenerator:
    """
    Generates W3C SSML 1.1 documents from ProsodyTarget or EmotionalIntent.

    Converts internal AXIOM VOX prosody representation to
    portable SSML that can be used with various TTS backends.
    """

    def generate(
        self,
        text: str,
        prosody_target: "ProsodyTarget",
        include_speak_wrapper: bool = True,
        xml_declaration: bool = False,
    ) -> str:
        """
        Generate SSML document from text and ProsodyTarget.

        Args:
            text: Plain text to convert
            prosody_target: ProsodyTarget with prosody parameters
            include_speak_wrapper: Wrap in <speak> tags
            xml_declaration: Include XML declaration

        Returns:
            SSML markup string
        """
        words = text.split()
        if not words:
            return '<speak></speak>' if include_speak_wrapper else ''

        result_parts: List[str] = []

        # Track which words need special handling
        emphasis_set = set(getattr(prosody_target, 'emphasis_words', []))
        pause_map = getattr(prosody_target, 'pause_locations', {})
        breath_set = set(getattr(prosody_target, 'breath_locations', []))

        # Build word-by-word with SSML annotations
        for i, word in enumerate(words):
            # Check for breath/short pause before word
            if i in breath_set:
                result_parts.append('<break time="200ms" strength="weak"/>')

            # Handle emphasis
            if i in emphasis_set:
                result_parts.append(f'<emphasis level="moderate">{self._escape(word)}</emphasis>')
            else:
                result_parts.append(self._escape(word))

            # Check for pause after word
            if i in pause_map:
                pause_ms = int(pause_map[i] * 1000)
                if pause_ms > 0:
                    result_parts.append(f'<break time="{pause_ms}ms"/>')

        inner_text = ' '.join(result_parts)

        # Wrap with prosody if parameters differ from default
        prosody_attrs = self._build_prosody_attrs(prosody_target)
        if prosody_attrs:
            inner_text = f'<prosody {prosody_attrs}>{inner_text}</prosody>'

        # Wrap with speak
        if include_speak_wrapper:
            inner_text = f'<speak>{inner_text}</speak>'

        # Add XML declaration
        if xml_declaration:
            inner_text = '<?xml version="1.0" encoding="UTF-8"?>\n' + inner_text

        return inner_text

    def generate_from_emotional_intent(
        self,
        text: str,
        intent: "EmotionalIntent",
        include_speak_wrapper: bool = True,
    ) -> str:
        """
        Generate SSML directly from EmotionalIntent.

        Converts 5D emotional space to SSML prosody parameters.
        """
        attrs: List[str] = []

        # Map emotional dimensions to SSML parameters
        rate = self._valence_arousal_to_rate(intent.valence, intent.arousal)
        if rate != "medium":
            attrs.append(f'rate="{rate}"')

        pitch = self._dominance_to_pitch(intent.dominance)
        if pitch:
            attrs.append(f'pitch="{pitch}"')

        volume = self._arousal_to_volume(intent.arousal)
        if volume != "medium":
            attrs.append(f'volume="{volume}"')

        escaped_text = self._escape(text)

        if attrs:
            inner = f'<prosody {" ".join(attrs)}>{escaped_text}</prosody>'
        else:
            inner = escaped_text

        if include_speak_wrapper:
            return f'<speak>{inner}</speak>'
        return inner

    def generate_simple(
        self,
        text: str,
        rate: Optional[float] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        pauses: Optional[Dict[int, float]] = None,
        emphases: Optional[List[int]] = None,
    ) -> str:
        """
        Generate SSML with explicit parameters.

        Args:
            text: Plain text
            rate: Speaking rate multiplier (1.0 = normal)
            pitch: Pitch in semitones
            volume: Volume multiplier (1.0 = normal)
            pauses: Word index -> pause seconds
            emphases: List of word indices to emphasize
        """
        words = text.split()
        if not words:
            return '<speak></speak>'

        pauses = pauses or {}
        emphases = emphases or []
        emphasis_set = set(emphases)

        parts: List[str] = []
        for i, word in enumerate(words):
            if i in emphasis_set:
                parts.append(f'<emphasis level="moderate">{self._escape(word)}</emphasis>')
            else:
                parts.append(self._escape(word))

            if i in pauses:
                ms = int(pauses[i] * 1000)
                if ms > 0:
                    parts.append(f'<break time="{ms}ms"/>')

        inner = ' '.join(parts)

        # Build prosody wrapper
        prosody_attrs: List[str] = []
        if rate is not None and rate != 1.0:
            pct = int(rate * 100)
            prosody_attrs.append(f'rate="{pct}%"')
        if pitch is not None and pitch != 0:
            sign = '+' if pitch > 0 else ''
            # Format as integer if whole number, otherwise keep decimal
            pitch_val = int(pitch) if pitch == int(pitch) else pitch
            prosody_attrs.append(f'pitch="{sign}{pitch_val}st"')
        if volume is not None and volume != 1.0:
            if volume > 1.2:
                prosody_attrs.append('volume="loud"')
            elif volume < 0.8:
                prosody_attrs.append('volume="soft"')

        if prosody_attrs:
            inner = f'<prosody {" ".join(prosody_attrs)}>{inner}</prosody>'

        return f'<speak>{inner}</speak>'

    def _build_prosody_attrs(self, target: "ProsodyTarget") -> str:
        """Build prosody element attributes from ProsodyTarget."""
        attrs: List[str] = []

        # Rate
        rate = getattr(target, 'speaking_rate', 1.0)
        if rate != 1.0:
            rate_pct = int(rate * 100)
            attrs.append(f'rate="{rate_pct}%"')

        # Pitch
        pitch = getattr(target, 'pitch_base', 0)
        if pitch != 0:
            sign = '+' if pitch > 0 else ''
            # Format as integer if whole number, otherwise keep decimal
            pitch_val = int(pitch) if pitch == int(pitch) else pitch
            attrs.append(f'pitch="{sign}{pitch_val}st"')

        # Volume (map energy to volume)
        energy = getattr(target, 'energy', 0.5)
        if energy > 0.7:
            attrs.append('volume="loud"')
        elif energy < 0.3:
            attrs.append('volume="soft"')

        return ' '.join(attrs)

    def _valence_arousal_to_rate(self, valence: float, arousal: float) -> str:
        """Map valence/arousal to speaking rate keyword."""
        # High arousal + positive valence = faster
        # Low arousal + negative valence = slower
        combined = (arousal + (valence + 1) / 2) / 2
        if combined > 0.7:
            return "fast"
        elif combined > 0.55:
            return "medium"
        elif combined > 0.3:
            return "slow"
        else:
            return "x-slow"

    def _dominance_to_pitch(self, dominance: float) -> Optional[str]:
        """Map dominance to pitch."""
        if dominance > 0.7:
            return "-2st"  # Lower pitch = more dominant
        elif dominance < 0.3:
            return "+2st"  # Higher pitch = less dominant
        return None

    def _arousal_to_volume(self, arousal: float) -> str:
        """Map arousal to volume keyword."""
        if arousal > 0.8:
            return "loud"
        elif arousal > 0.6:
            return "medium"
        elif arousal > 0.3:
            return "soft"
        return "x-soft"

    def _escape(self, text: str) -> str:
        """Escape XML special characters."""
        return (
            text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;')
        )


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  AXIOM VOX SSML Parser & Generator Demo")
    print("=" * 70)

    parser = SSMLParser()
    generator = SSMLGenerator()

    # Test 1: Basic parsing
    print("\n1. Basic SSML parsing:")
    ssml1 = '<speak>Hello world. How are you?</speak>'
    doc1, warns1 = parser.parse(ssml1)
    print(f"   Input: {ssml1}")
    print(f"   Plain text: '{doc1.plain_text}'")
    print(f"   Words: {doc1.word_list}")

    # Test 2: Break elements
    print("\n2. Break elements:")
    ssml2 = '<speak>Hello<break time="500ms"/> world!</speak>'
    doc2, warns2 = parser.parse(ssml2)
    print(f"   Input: {ssml2}")
    print(f"   Plain text: '{doc2.plain_text}'")
    print(f"   Breaks: {[(b.word_index, b.time_ms) for b in doc2.breaks]}")
    print(f"   Pause locations: {doc2.pause_locations}")

    # Test 3: Emphasis
    print("\n3. Emphasis elements:")
    ssml3 = '<speak>This is <emphasis level="strong">very important</emphasis> text.</speak>'
    doc3, warns3 = parser.parse(ssml3)
    print(f"   Input: {ssml3}")
    print(f"   Plain text: '{doc3.plain_text}'")
    print(f"   Emphases: {[(e.text, e.level) for e in doc3.emphases]}")
    print(f"   Emphasis word indices: {doc3.emphasis_words}")

    # Test 4: Prosody
    print("\n4. Prosody elements:")
    ssml4 = '<speak><prosody rate="slow" pitch="+2st">Slow and high text</prosody></speak>'
    doc4, warns4 = parser.parse(ssml4)
    print(f"   Input: {ssml4}")
    print(f"   Plain text: '{doc4.plain_text}'")
    print(f"   Prosody spans: {[(p.rate, p.pitch) for p in doc4.prosody_spans]}")

    # Test 5: Substitution
    print("\n5. Substitution elements:")
    ssml5 = '<speak>The <sub alias="World Wide Web Consortium">W3C</sub> defines standards.</speak>'
    doc5, warns5 = parser.parse(ssml5)
    print(f"   Input: {ssml5}")
    print(f"   Plain text: '{doc5.plain_text}'")
    print(f"   Substitutions: {[(s.original, s.alias) for s in doc5.substitutions]}")

    # Test 6: Say-as
    print("\n6. Say-as elements:")
    ssml6 = '<speak>Call <say-as interpret-as="telephone">+1-800-555-1234</say-as></speak>'
    doc6, warns6 = parser.parse(ssml6)
    print(f"   Input: {ssml6}")
    print(f"   Plain text: '{doc6.plain_text}'")
    print(f"   Say-as: {[(s.text, s.interpret_as) for s in doc6.say_as_spans]}")

    # Test 7: SSML generation
    print("\n7. SSML generation:")
    generated = generator.generate_simple(
        text="Hello beautiful world today",
        rate=0.9,
        pitch=1.5,
        pauses={1: 0.3},
        emphases=[0, 2],
    )
    print(f"   Generated: {generated}")

    # Test 8: Plain text fallback
    print("\n8. Plain text fallback:")
    plain = "Just plain text without any markup"
    doc8, warns8 = parser.parse(plain)
    print(f"   Input: {plain}")
    print(f"   Plain text: '{doc8.plain_text}'")
    print(f"   Words: {len(doc8.word_list)}")

    # Test 9: Multi-voice (v0.9.0)
    print("\n9. Multi-voice elements (v0.9.0):")
    ssml9 = '''<speak>
        <voice axiom-voice="professional">Welcome to the presentation.</voice>
        <voice axiom-voice="expert" emotion="confident">Let me explain the details.</voice>
        <voice name="calm">And now for the conclusion.</voice>
    </speak>'''
    doc9, warns9 = parser.parse(ssml9)
    print(f"   Plain text: '{doc9.plain_text}'")
    print(f"   Voice spans: {len(doc9.voice_spans)}")
    for v in doc9.voice_spans:
        print(f"      - voice_id={v.voice_id}, name={v.name}, emotion={v.emotion}")
        print(f"        text: '{v.text}'")
    print(f"   Is multi-voice: {doc9.is_multi_voice}")
    print(f"   Voices used: {doc9.voices_used}")

    # Test 10: Voice with character mapping
    print("\n10. Voice with character attribute:")
    ssml10 = '''<speak>
        <voice character="Dr. Smith" emotion="enthusiastic">Fascinating discovery!</voice>
        <voice character="Host" axiom-voice="announcer">Thank you, Doctor.</voice>
    </speak>'''
    doc10, warns10 = parser.parse(ssml10)
    print(f"   Plain text: '{doc10.plain_text}'")
    for v in doc10.voice_spans:
        print(f"      - character={v.character}, voice_id={v.voice_id}, emotion={v.emotion}")
        print(f"        resolved_voice: {v.get_resolved_voice_id()}")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
