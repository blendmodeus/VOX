"""
AXIOM VOX Multi-Voice Synthesizer
----------------------------------

Synthesis engine for multi-voice audio with dialogue support.

Enables seamless synthesis across multiple voices with:
- DialogueScript input (structured API)
- SSML with <voice> tags
- Automatic voice transitions
- Character-to-voice mapping
- Per-voice governance

v0.9.0: Multi-voice Synthesis

Usage:
    from axiom_vox.multi_voice_synthesizer import MultiVoiceSynthesizer
    from axiom_vox.multi_voice import DialogueScript, DialogueLine

    synthesizer = MultiVoiceSynthesizer()

    # From DialogueScript
    script = DialogueScript(lines=[
        DialogueLine(text="Welcome!", voice_id="announcer"),
        DialogueLine(text="Let me explain.", voice_id="expert"),
    ])
    result = synthesizer.synthesize_script(script)

    # From SSML
    ssml = '''<speak>
        <voice axiom-voice="professional">Hello.</voice>
        <voice axiom-voice="calm">Goodbye.</voice>
    </speak>'''
    result = synthesizer.synthesize_ssml(ssml)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from axiom_vox.multi_voice import (
    DialogueLine,
    DialogueScript,
    MultiVoiceSegment,
    MultiVoiceSynthesisResult,
    TransitionStyle,
    VoiceSwitch,
    VoiceSwitchType,
)
from axiom_vox.transition_processor import (
    TransitionConfig,
    TransitionProcessor,
    TransitionResult,
)
from axiom_vox.character_registry import (
    CharacterRegistry,
    get_character_registry,
)

if TYPE_CHECKING:
    from axiom_vox.synthesis import VoxSynthesizer, VoiceConfig, SynthesisResult
    from axiom_vox.vox_governor import VoxGovernor
    from axiom_vox.ssml import SSMLDocument

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MultiVoiceConfig:
    """Configuration for multi-voice synthesis."""

    # Default transition style
    default_transition: TransitionStyle = TransitionStyle.BREATH_PAUSE

    # Transition timing (ms)
    breath_pause_ms: int = 200
    crossfade_ms: int = 150
    silence_ms: int = 100

    # Governance
    pre_validate_voices: bool = True  # Validate all voices before synthesis
    max_switches_per_minute: int = 30  # Rate limit for voice switches

    # Audio output
    sample_rate: int = 24000
    normalize_volume: bool = True
    target_db: float = -3.0

    # Performance
    parallel_sentence_synthesis: bool = False  # Synthesize sentences in parallel
    cache_voice_configs: bool = True


# ============================================================================
# MULTI-VOICE SYNTHESIS RESULT
# ============================================================================

@dataclass
class SegmentSynthesisResult:
    """Result from synthesizing a single segment."""
    segment_index: int
    voice_id: str
    text: str
    audio_bytes: bytes
    duration_ms: float
    success: bool
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return len(self.audio_bytes) == 0


# ============================================================================
# MULTI-VOICE SYNTHESIZER
# ============================================================================

class MultiVoiceSynthesizer:
    """
    Multi-voice synthesis engine.

    Orchestrates synthesis across multiple voices with
    automatic transitions and governance integration.
    """

    def __init__(
        self,
        synthesizer: Optional["VoxSynthesizer"] = None,
        governor: Optional["VoxGovernor"] = None,
        character_registry: Optional[CharacterRegistry] = None,
        config: Optional[MultiVoiceConfig] = None,
    ):
        """
        Initialize multi-voice synthesizer.

        Args:
            synthesizer: Base synthesizer (lazy-loaded if None)
            governor: Governor for content checks (optional)
            character_registry: Character-to-voice mapping
            config: Multi-voice configuration
        """
        self._synthesizer = synthesizer
        self._governor = governor
        self._character_registry = character_registry or get_character_registry()
        self.config = config or MultiVoiceConfig()

        self._transition_processor = TransitionProcessor(
            sample_rate=self.config.sample_rate
        )

        # Voice config cache
        self._voice_config_cache: Dict[str, "VoiceConfig"] = {}

        # Statistics
        self._total_syntheses = 0
        self._total_switches = 0
        self._last_switch_times: List[float] = []

    @property
    def synthesizer(self) -> "VoxSynthesizer":
        """Get or create synthesizer."""
        if self._synthesizer is None:
            from axiom_vox.synthesis import get_synthesizer
            self._synthesizer = get_synthesizer()
        return self._synthesizer

    @property
    def governor(self) -> Optional["VoxGovernor"]:
        """Get governor if available."""
        if self._governor is None:
            try:
                from axiom_vox.vox_governor import get_governor
                self._governor = get_governor()
            except Exception:
                pass
        return self._governor

    def synthesize_script(
        self,
        script: DialogueScript,
        output_format: str = "wav",
    ) -> MultiVoiceSynthesisResult:
        """
        Synthesize a dialogue script.

        Args:
            script: DialogueScript with lines and transitions
            output_format: Audio output format

        Returns:
            MultiVoiceSynthesisResult with combined audio
        """
        start_time = time.time()
        segments: List[MultiVoiceSegment] = []
        voice_switches: List[VoiceSwitch] = []
        audio_parts: List[bytes] = []
        errors: List[str] = []

        # Pre-validate all voices if configured
        if self.config.pre_validate_voices:
            validation_errors = self._validate_voices(script.voices_used)
            if validation_errors:
                errors.extend(validation_errors)
                # Continue with valid voices

        current_voice: Optional[str] = None
        current_offset_ms = 0.0

        for idx, line in enumerate(script.lines):
            # Resolve voice from character if needed
            voice_id = self._resolve_voice_id(line)

            # Check for voice switch
            if current_voice is not None and voice_id != current_voice:
                # Record switch
                switch_type = VoiceSwitchType.CHARACTER_CHANGE
                if line.character_name:
                    switch_type = VoiceSwitchType.CHARACTER_CHANGE
                elif hasattr(line, 'is_quote') and line.is_quote:
                    switch_type = VoiceSwitchType.DIALOGUE

                voice_switches.append(VoiceSwitch(
                    from_voice=current_voice,
                    to_voice=voice_id,
                    switch_type=switch_type,
                    transition_style=script.default_transition,
                    word_index=idx,
                    character_from=None,
                    character_to=line.character_name,
                ))

                # Check rate limit
                if not self._check_switch_rate():
                    errors.append(
                        f"Voice switch rate limit exceeded at line {idx}"
                    )
                    continue

                # Apply pause before if specified
                if line.pause_before_ms > 0:
                    pause_audio = self._transition_processor.generate_silence(
                        line.pause_before_ms
                    )
                    audio_parts.append(pause_audio)
                    current_offset_ms += line.pause_before_ms

                # Apply transition
                transition = self._create_transition(
                    script.default_transition,
                    previous_audio=audio_parts[-1] if audio_parts else None,
                )
                if transition and not transition.is_empty:
                    audio_parts.append(transition.audio_bytes)
                    current_offset_ms += transition.duration_ms

            current_voice = voice_id

            # Synthesize segment
            segment_result = self._synthesize_segment(
                text=line.text,
                voice_id=voice_id,
                emotion=line.emotion,
                output_format=output_format,
            )

            if not segment_result.success:
                errors.append(segment_result.error or f"Failed to synthesize line {idx}")
                continue

            # Record segment
            segments.append(MultiVoiceSegment(
                text=line.text,
                voice_id=voice_id,
                character_name=line.character_name,
                start_ms=current_offset_ms,
                end_ms=current_offset_ms + segment_result.duration_ms,
                audio_bytes=segment_result.audio_bytes,
            ))

            audio_parts.append(segment_result.audio_bytes)
            current_offset_ms += segment_result.duration_ms

            # Apply pause after if specified
            if line.pause_after_ms > 0:
                pause_audio = self._transition_processor.generate_silence(
                    line.pause_after_ms
                )
                audio_parts.append(pause_audio)
                current_offset_ms += line.pause_after_ms

        # Combine all audio
        combined_audio = b"".join(audio_parts)

        # Normalize volume if configured
        if self.config.normalize_volume and combined_audio:
            combined_audio = self._transition_processor.normalize_volume(
                combined_audio,
                target_db=self.config.target_db,
            )

        # Calculate stats
        total_duration_ms = current_offset_ms
        synthesis_time = time.time() - start_time

        self._total_syntheses += 1

        return MultiVoiceSynthesisResult(
            success=len(errors) == 0,
            audio_bytes=combined_audio,
            total_duration_ms=total_duration_ms,
            segments=segments,
            voice_switches=voice_switches,
            voices_used=list(script.voices_used),
            synthesis_time_ms=synthesis_time * 1000,
            errors=errors if errors else None,
        )

    def synthesize_ssml(
        self,
        ssml: str,
        default_voice_id: str = "professional",
        output_format: str = "wav",
    ) -> MultiVoiceSynthesisResult:
        """
        Synthesize SSML with multi-voice support.

        Args:
            ssml: SSML markup with <voice> tags
            default_voice_id: Default voice for untagged content
            output_format: Audio output format

        Returns:
            MultiVoiceSynthesisResult with combined audio
        """
        from axiom_vox.ssml import SSMLParser

        parser = SSMLParser()
        doc, warnings = parser.parse(ssml)

        if warnings:
            logger.warning(f"SSML parse warnings: {warnings}")

        # Check if multi-voice
        if not doc.is_multi_voice:
            # Single voice - use standard synthesis
            result = self._synthesize_segment(
                text=doc.plain_text,
                voice_id=default_voice_id,
                output_format=output_format,
            )
            return MultiVoiceSynthesisResult(
                success=result.success,
                audio_bytes=result.audio_bytes,
                total_duration_ms=result.duration_ms,
                segments=[MultiVoiceSegment(
                    text=doc.plain_text,
                    voice_id=default_voice_id,
                    start_ms=0,
                    end_ms=result.duration_ms,
                    audio_bytes=result.audio_bytes,
                )],
                voice_switches=[],
                voices_used=[default_voice_id],
                synthesis_time_ms=0,
            )

        # Convert SSML voice spans to DialogueScript
        script = self._ssml_to_script(doc, default_voice_id)

        return self.synthesize_script(script, output_format)

    async def synthesize_script_stream(
        self,
        script: DialogueScript,
        output_format: str = "wav",
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Stream multi-voice synthesis.

        Yields tuples of (message_type, data):
        - ("started", {"voices": [...], "lines": n})
        - ("voice_switch", {"from": ..., "to": ...})
        - ("segment", {"voice_id": ..., "text": ...})
        - ("audio", audio_bytes)
        - ("completed", {"duration_ms": ..., "voices": ...})
        - ("error", {"message": ...})

        Args:
            script: DialogueScript
            output_format: Audio format

        Yields:
            (message_type, data) tuples
        """
        yield ("started", {
            "voices": list(script.voices_used),
            "lines": len(script.lines),
        })

        current_voice: Optional[str] = None

        for idx, line in enumerate(script.lines):
            voice_id = self._resolve_voice_id(line)

            # Voice switch event
            if current_voice is not None and voice_id != current_voice:
                yield ("voice_switch", {
                    "from": current_voice,
                    "to": voice_id,
                    "line_index": idx,
                })

                # Generate transition
                transition = self._create_transition(script.default_transition)
                if transition and not transition.is_empty:
                    yield ("audio", transition.audio_bytes)

            current_voice = voice_id

            # Segment event
            yield ("segment", {
                "voice_id": voice_id,
                "text": line.text,
                "character": line.character_name,
                "line_index": idx,
            })

            # Synthesize and yield audio
            result = self._synthesize_segment(
                text=line.text,
                voice_id=voice_id,
                emotion=line.emotion,
                output_format=output_format,
            )

            if result.success:
                yield ("audio", result.audio_bytes)
            else:
                yield ("error", {"message": result.error, "line_index": idx})

            # Small delay for async behavior
            await asyncio.sleep(0.001)

        yield ("completed", {
            "voices": list(script.voices_used),
            "lines": len(script.lines),
        })

    def _resolve_voice_id(self, line: DialogueLine) -> str:
        """Resolve voice ID from line, using character registry if needed."""
        if line.voice_id:
            return line.voice_id

        if line.character_name:
            # Look up in character registry
            voice = self._character_registry.get_voice(line.character_name)
            if voice:
                return voice

            # Auto-assign voice
            return self._character_registry.get_or_assign(line.character_name)

        # Default
        return "professional"

    def _synthesize_segment(
        self,
        text: str,
        voice_id: str,
        emotion: Optional[str] = None,
        output_format: str = "wav",
    ) -> SegmentSynthesisResult:
        """Synthesize a single segment."""
        from axiom_vox.synthesis import VoiceConfig, AudioFormat

        start_time = time.time()

        try:
            # Get or create voice config
            voice_config = self._get_voice_config(voice_id, emotion)

            # Convert format
            fmt = AudioFormat(output_format)

            # Synthesize
            result = self.synthesizer.synthesize(
                text=text,
                voice=voice_config,
                output_format=fmt,
            )

            duration_ms = (result.duration_seconds or 0) * 1000

            return SegmentSynthesisResult(
                segment_index=0,
                voice_id=voice_id,
                text=text,
                audio_bytes=result.audio_data or b"",
                duration_ms=duration_ms,
                success=result.success,
                error=result.error,
            )

        except Exception as e:
            logger.error(f"Segment synthesis failed: {e}")
            return SegmentSynthesisResult(
                segment_index=0,
                voice_id=voice_id,
                text=text,
                audio_bytes=b"",
                duration_ms=0,
                success=False,
                error=str(e),
            )

    def _get_voice_config(
        self,
        voice_id: str,
        emotion: Optional[str] = None,
    ) -> "VoiceConfig":
        """Get or create voice configuration."""
        from axiom_vox.synthesis import VoiceConfig

        cache_key = f"{voice_id}:{emotion or 'default'}"

        if self.config.cache_voice_configs and cache_key in self._voice_config_cache:
            return self._voice_config_cache[cache_key]

        # Check character registry for overrides
        mapping = self._character_registry.get(voice_id)

        config = VoiceConfig(
            voice_id=voice_id,
            emotion=emotion or (mapping.default_emotion if mapping else None),
            speaking_rate=mapping.speaking_rate if mapping and mapping.speaking_rate else 1.0,
            pitch=mapping.pitch if mapping and mapping.pitch else 0.0,
            volume=mapping.volume if mapping and mapping.volume else 1.0,
        )

        if self.config.cache_voice_configs:
            self._voice_config_cache[cache_key] = config

        return config

    def _create_transition(
        self,
        style: TransitionStyle,
        previous_audio: Optional[bytes] = None,
        next_audio: Optional[bytes] = None,
    ) -> Optional[TransitionResult]:
        """Create a transition between voice segments."""
        config = TransitionConfig(
            style=style,
            duration_ms=self._get_transition_duration(style),
        )

        return self._transition_processor.create_transition(
            style=style,
            config=config,
            previous_audio=previous_audio,
            next_audio=next_audio,
        )

    def _get_transition_duration(self, style: TransitionStyle) -> int:
        """Get transition duration based on style and config."""
        if style == TransitionStyle.BREATH_PAUSE:
            return self.config.breath_pause_ms
        elif style == TransitionStyle.CROSSFADE:
            return self.config.crossfade_ms
        elif style == TransitionStyle.SILENCE:
            return self.config.silence_ms
        return 0

    def _validate_voices(self, voice_ids: Set[str]) -> List[str]:
        """Validate all voices before synthesis."""
        errors = []

        governor = self.governor
        if not governor:
            return errors

        for voice_id in voice_ids:
            # Check voice boundaries
            try:
                from axiom_vox.voice_boundaries import VoiceBoundaryChecker
                checker = VoiceBoundaryChecker()
                if not checker.is_voice_allowed(voice_id):
                    errors.append(f"Voice '{voice_id}' is not allowed")
            except ImportError:
                pass  # Voice boundaries not available
            except Exception as e:
                logger.warning(f"Voice validation error for {voice_id}: {e}")

        return errors

    def _check_switch_rate(self) -> bool:
        """Check if voice switch rate is within limits."""
        now = time.time()

        # Clean old entries (older than 60 seconds)
        self._last_switch_times = [
            t for t in self._last_switch_times
            if now - t < 60
        ]

        # Check limit
        if len(self._last_switch_times) >= self.config.max_switches_per_minute:
            return False

        self._last_switch_times.append(now)
        self._total_switches += 1
        return True

    def _ssml_to_script(
        self,
        doc: "SSMLDocument",
        default_voice_id: str,
    ) -> DialogueScript:
        """Convert SSML document to DialogueScript."""
        lines: List[DialogueLine] = []

        for voice_span in doc.voice_spans:
            voice_id = voice_span.get_resolved_voice_id() or default_voice_id

            lines.append(DialogueLine(
                text=voice_span.text,
                voice_id=voice_id,
                character_name=voice_span.character,
                emotion=voice_span.emotion,
            ))

        return DialogueScript(
            lines=lines,
            default_transition=TransitionStyle.BREATH_PAUSE,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get synthesizer statistics."""
        return {
            "total_syntheses": self._total_syntheses,
            "total_voice_switches": self._total_switches,
            "cached_voice_configs": len(self._voice_config_cache),
            "registered_characters": len(self._character_registry.list_character_names()),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_default_synthesizer: Optional[MultiVoiceSynthesizer] = None


def get_multi_voice_synthesizer() -> MultiVoiceSynthesizer:
    """Get or create default multi-voice synthesizer."""
    global _default_synthesizer
    if _default_synthesizer is None:
        _default_synthesizer = MultiVoiceSynthesizer()
    return _default_synthesizer


def synthesize_dialogue(
    lines: List[Dict[str, Any]],
    default_transition: str = "breath_pause",
) -> MultiVoiceSynthesisResult:
    """
    Quick function to synthesize dialogue.

    Args:
        lines: List of dicts with 'text', 'voice_id', optional 'character', 'emotion'
        default_transition: Transition style name

    Returns:
        MultiVoiceSynthesisResult
    """
    synthesizer = get_multi_voice_synthesizer()

    dialogue_lines = [
        DialogueLine(
            text=line["text"],
            voice_id=line.get("voice_id", "professional"),
            character_name=line.get("character"),
            emotion=line.get("emotion"),
        )
        for line in lines
    ]

    transition = TransitionStyle(default_transition)

    script = DialogueScript(
        lines=dialogue_lines,
        default_transition=transition,
    )

    return synthesizer.synthesize_script(script)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  AXIOM VOX Multi-Voice Synthesizer Demo")
    print("=" * 70)

    synthesizer = MultiVoiceSynthesizer()

    # Demo 1: DialogueScript
    print("\n1. DialogueScript Synthesis:")
    script = DialogueScript(
        lines=[
            DialogueLine(
                text="Welcome to our presentation!",
                voice_id="announcer",
                character_name="Host",
            ),
            DialogueLine(
                text="Today I'll explain the technology.",
                voice_id="expert",
                character_name="Dr. Smith",
                emotion="confident",
            ),
            DialogueLine(
                text="Let's begin with the basics.",
                voice_id="guide",
                character_name="Instructor",
            ),
        ],
        default_transition=TransitionStyle.BREATH_PAUSE,
    )

    print(f"   Lines: {len(script.lines)}")
    print(f"   Voices: {script.voices_used}")

    result = synthesizer.synthesize_script(script)
    print(f"   Success: {result.success}")
    print(f"   Duration: {result.total_duration_ms:.1f}ms")
    print(f"   Segments: {len(result.segments)}")
    print(f"   Voice switches: {len(result.voice_switches)}")

    # Demo 2: SSML Multi-voice
    print("\n2. SSML Multi-voice Synthesis:")
    ssml = '''<speak>
        <voice axiom-voice="professional">Welcome to the demo.</voice>
        <voice axiom-voice="calm" emotion="relaxed">This is the calm voice.</voice>
        <voice axiom-voice="expert">And this is the expert.</voice>
    </speak>'''

    result2 = synthesizer.synthesize_ssml(ssml)
    print(f"   Success: {result2.success}")
    print(f"   Voices used: {result2.voices_used}")
    print(f"   Segments: {len(result2.segments)}")

    # Demo 3: Quick function
    print("\n3. Quick Dialogue Function:")
    result3 = synthesize_dialogue([
        {"text": "How are you?", "voice_id": "conversational"},
        {"text": "I'm doing well!", "voice_id": "casual", "emotion": "happy"},
    ])
    print(f"   Success: {result3.success}")
    print(f"   Segments: {len(result3.segments)}")

    # Stats
    print("\n4. Synthesizer Stats:")
    stats = synthesizer.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)
