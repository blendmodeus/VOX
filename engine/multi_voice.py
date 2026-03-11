"""
AXIOM VOX Multi-voice Synthesis Data Models
--------------------------------------------

Core data structures for multi-voice TTS synthesis:
- DialogueLine: Single line with voice assignment
- DialogueScript: Complete multi-voice script
- VoiceSwitch: Voice transition point
- MultiVoiceSegment: Processing unit for synthesis

v0.9.0: Multi-voice Synthesis

Usage:
    from axiom_vox.multi_voice import DialogueScript, DialogueLine

    script = DialogueScript(lines=[
        DialogueLine(text="Welcome!", voice_id="announcer"),
        DialogueLine(text="Let me explain.", voice_id="expert"),
    ])

    print(f"Voices: {script.voices_used}")
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from axiom_vox.prosody_guardrails import EmotionalIntent


# ============================================================================
# ENUMS
# ============================================================================


class TransitionStyle(str, Enum):
    """How to transition between voices."""
    CROSSFADE = "crossfade"        # Audio overlap with volume curves (~150ms)
    BREATH_PAUSE = "breath_pause"  # Natural breath pause (~200ms) - default
    SILENCE = "silence"            # Clean gap
    IMMEDIATE = "immediate"        # Direct cut (not recommended)


class VoiceSwitchType(str, Enum):
    """When a voice switch occurs."""
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH = "paragraph"
    IMMEDIATE = "immediate"  # Mid-sentence (not recommended)
    DIALOGUE_TURN = "dialogue_turn"


# ============================================================================
# DIALOGUE MODELS
# ============================================================================


@dataclass
class DialogueLine:
    """
    A single line of dialogue with voice assignment.

    Represents one speaker's contribution in a multi-voice script.
    """
    text: str
    voice_id: str
    character_name: Optional[str] = None
    emotion: Optional[str] = None  # Emotion preset name
    emotion_intent: Optional[EmotionalIntent] = None
    index: int = 0

    # Timing hints
    pause_before_ms: int = 0  # Pause before this line
    pause_after_ms: int = 0   # Pause after this line

    # Governance results (populated after check)
    governance_passed: bool = False
    governed_text: Optional[str] = None
    governance_warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate after initialization."""
        if not self.text or not self.text.strip():
            raise ValueError("DialogueLine text cannot be empty")
        if not self.voice_id or not self.voice_id.strip():
            raise ValueError("DialogueLine voice_id cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "voice_id": self.voice_id,
            "character_name": self.character_name,
            "emotion": self.emotion,
            "index": self.index,
            "pause_before_ms": self.pause_before_ms,
            "pause_after_ms": self.pause_after_ms,
            "governance_passed": self.governance_passed,
            "governed_text": self.governed_text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueLine":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            voice_id=data["voice_id"],
            character_name=data.get("character_name"),
            emotion=data.get("emotion"),
            index=data.get("index", 0),
            pause_before_ms=data.get("pause_before_ms", 0),
            pause_after_ms=data.get("pause_after_ms", 0),
        )


@dataclass
class VoiceSwitch:
    """
    Represents a voice transition point.

    Tracks when and how voices change in a multi-voice stream.
    """
    from_voice_id: Optional[str]
    to_voice_id: str
    position: int  # Line index where switch occurs
    switch_type: VoiceSwitchType = VoiceSwitchType.DIALOGUE_TURN
    transition_style: TransitionStyle = TransitionStyle.BREATH_PAUSE
    transition_duration_ms: int = 200

    # Governance tracking
    governance_checked: bool = False
    governance_approved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_voice_id": self.from_voice_id,
            "to_voice_id": self.to_voice_id,
            "position": self.position,
            "switch_type": self.switch_type.value,
            "transition_style": self.transition_style.value,
            "transition_duration_ms": self.transition_duration_ms,
            "governance_approved": self.governance_approved,
        }


@dataclass
class DialogueScript:
    """
    Complete multi-voice script.

    Contains multiple dialogue lines with voice assignments
    and controls for transitions between voices.
    """
    lines: List[DialogueLine]

    # Metadata
    title: Optional[str] = None
    description: Optional[str] = None
    script_id: str = field(default_factory=lambda: f"script_{uuid.uuid4().hex[:12]}")

    # Transition settings
    default_transition: TransitionStyle = TransitionStyle.BREATH_PAUSE
    default_pause_ms: int = 200

    # Computed after construction
    _voice_switches: Optional[List[VoiceSwitch]] = field(default=None, repr=False)

    def __post_init__(self):
        """Index lines after initialization."""
        for i, line in enumerate(self.lines):
            line.index = i

    @property
    def voices_used(self) -> Set[str]:
        """Get set of all voice IDs used in script."""
        return {line.voice_id for line in self.lines}

    @property
    def voice_count(self) -> int:
        """Number of unique voices."""
        return len(self.voices_used)

    @property
    def line_count(self) -> int:
        """Number of dialogue lines."""
        return len(self.lines)

    @property
    def total_text_length(self) -> int:
        """Total character count across all lines."""
        return sum(len(line.text) for line in self.lines)

    def get_voice_switches(self) -> List[VoiceSwitch]:
        """
        Extract all voice transition points.

        Returns cached result if already computed.
        """
        if self._voice_switches is not None:
            return self._voice_switches

        switches = []
        prev_voice = None

        for i, line in enumerate(self.lines):
            if line.voice_id != prev_voice:
                switches.append(VoiceSwitch(
                    from_voice_id=prev_voice,
                    to_voice_id=line.voice_id,
                    position=i,
                    switch_type=VoiceSwitchType.DIALOGUE_TURN,
                    transition_style=self.default_transition,
                    transition_duration_ms=self.default_pause_ms,
                ))
                prev_voice = line.voice_id

        self._voice_switches = switches
        return switches

    @property
    def switch_count(self) -> int:
        """Number of voice switches."""
        return len(self.get_voice_switches())

    def get_lines_for_voice(self, voice_id: str) -> List[DialogueLine]:
        """Get all lines for a specific voice."""
        return [line for line in self.lines if line.voice_id == voice_id]

    def get_character_lines(self, character_name: str) -> List[DialogueLine]:
        """Get all lines for a specific character."""
        return [
            line for line in self.lines
            if line.character_name and line.character_name.lower() == character_name.lower()
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "script_id": self.script_id,
            "title": self.title,
            "description": self.description,
            "lines": [line.to_dict() for line in self.lines],
            "voices_used": list(self.voices_used),
            "voice_count": self.voice_count,
            "line_count": self.line_count,
            "switch_count": self.switch_count,
            "default_transition": self.default_transition.value,
            "default_pause_ms": self.default_pause_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueScript":
        """Create from dictionary."""
        lines = [DialogueLine.from_dict(line) for line in data["lines"]]
        return cls(
            lines=lines,
            title=data.get("title"),
            description=data.get("description"),
            script_id=data.get("script_id", f"script_{uuid.uuid4().hex[:12]}"),
            default_transition=TransitionStyle(
                data.get("default_transition", "breath_pause")
            ),
            default_pause_ms=data.get("default_pause_ms", 200),
        )

    def validate(self) -> List[str]:
        """
        Validate script for common issues.

        Returns list of warning/error messages.
        """
        issues = []

        if not self.lines:
            issues.append("Script has no lines")
            return issues

        # Check for empty lines
        for i, line in enumerate(self.lines):
            if not line.text.strip():
                issues.append(f"Line {i} has empty text")

        # Check for unknown voices (informational)
        known_voices = {
            "professional", "conversational", "expert", "guide",
            "announcer", "calm", "casual", "corporate", "default",
        }
        for voice_id in self.voices_used:
            if not voice_id.startswith("clone_") and voice_id not in known_voices:
                issues.append(f"Voice '{voice_id}' may not be a registered voice")

        # Check for rapid switches
        if self.switch_count > self.line_count * 0.8:
            issues.append("High switch frequency may sound unnatural")

        return issues


# ============================================================================
# SYNTHESIS SEGMENT
# ============================================================================


@dataclass
class MultiVoiceSegment:
    """
    A segment of text with a single voice (internal processing unit).

    Used by MultiVoiceSynthesizer to break script into synthesis units.
    """
    text: str
    voice_id: str
    segment_index: int
    start_line_index: int
    end_line_index: int

    # Voice configuration
    emotion: Optional[str] = None
    emotion_intent: Optional[EmotionalIntent] = None

    # Timing
    pause_before_ms: int = 0
    pause_after_ms: int = 0

    # Sentence breakdown (populated during synthesis)
    sentence_count: int = 0
    sentences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "voice_id": self.voice_id,
            "segment_index": self.segment_index,
            "start_line_index": self.start_line_index,
            "end_line_index": self.end_line_index,
            "emotion": self.emotion,
            "pause_before_ms": self.pause_before_ms,
            "pause_after_ms": self.pause_after_ms,
            "sentence_count": self.sentence_count,
        }


# ============================================================================
# SYNTHESIS RESULT
# ============================================================================


@dataclass
class MultiVoiceSynthesisResult:
    """Result from multi-voice synthesis."""
    success: bool
    audio_data: Optional[bytes] = None
    duration_seconds: float = 0.0

    # Statistics
    lines_synthesized: int = 0
    voices_used: List[str] = field(default_factory=list)
    voice_switches: int = 0
    total_segments: int = 0

    # Per-voice stats
    voice_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Governance
    governance_reports: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Errors/warnings
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Timing
    synthesis_time_ms: float = 0.0
    first_audio_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes audio data)."""
        return {
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "lines_synthesized": self.lines_synthesized,
            "voices_used": self.voices_used,
            "voice_switches": self.voice_switches,
            "total_segments": self.total_segments,
            "voice_stats": self.voice_stats,
            "governance_reports": self.governance_reports,
            "error": self.error,
            "warnings": self.warnings,
            "synthesis_time_ms": self.synthesis_time_ms,
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def merge_consecutive_lines(lines: List[DialogueLine]) -> List[MultiVoiceSegment]:
    """
    Merge consecutive lines with same voice into segments.

    This reduces the number of synthesis calls and creates
    more natural speech flow.

    Args:
        lines: List of dialogue lines

    Returns:
        List of merged segments
    """
    if not lines:
        return []

    segments = []
    current_texts = []
    current_voice = lines[0].voice_id
    current_emotion = lines[0].emotion
    start_index = 0

    for i, line in enumerate(lines):
        if line.voice_id == current_voice and line.emotion == current_emotion:
            # Same voice and emotion, accumulate
            current_texts.append(line.text)
        else:
            # Voice or emotion changed, create segment
            segments.append(MultiVoiceSegment(
                text=" ".join(current_texts),
                voice_id=current_voice,
                segment_index=len(segments),
                start_line_index=start_index,
                end_line_index=i - 1,
                emotion=current_emotion,
            ))

            # Start new segment
            current_texts = [line.text]
            current_voice = line.voice_id
            current_emotion = line.emotion
            start_index = i

    # Don't forget the last segment
    if current_texts:
        segments.append(MultiVoiceSegment(
            text=" ".join(current_texts),
            voice_id=current_voice,
            segment_index=len(segments),
            start_line_index=start_index,
            end_line_index=len(lines) - 1,
            emotion=current_emotion,
        ))

    return segments


def parse_screenplay_format(text: str) -> DialogueScript:
    """
    Parse screenplay format into DialogueScript.

    Format:
        CHARACTER: Dialogue line here.
        OTHER_CHARACTER: Response here.

    Args:
        text: Screenplay formatted text

    Returns:
        DialogueScript with parsed lines
    """
    import re

    lines = []
    pattern = r'^([A-Z][A-Z\s]+):\s*(.+)$'

    for line_text in text.strip().split('\n'):
        line_text = line_text.strip()
        if not line_text:
            continue

        match = re.match(pattern, line_text)
        if match:
            character = match.group(1).strip()
            dialogue = match.group(2).strip()

            lines.append(DialogueLine(
                text=dialogue,
                voice_id="default",  # Will be assigned later
                character_name=character,
            ))
        else:
            # Non-dialogue line (stage direction, etc.) - skip or add as narrator
            if lines:
                # Append to previous line if exists
                lines[-1].text += " " + line_text

    return DialogueScript(lines=lines)


def parse_chat_format(text: str) -> DialogueScript:
    """
    Parse chat format into DialogueScript.

    Format:
        @alice: Hello there!
        @bob: Hi Alice!

    Args:
        text: Chat formatted text

    Returns:
        DialogueScript with parsed lines
    """
    import re

    lines = []
    pattern = r'^@(\w+):\s*(.+)$'

    for line_text in text.strip().split('\n'):
        line_text = line_text.strip()
        if not line_text:
            continue

        match = re.match(pattern, line_text)
        if match:
            character = match.group(1).strip()
            dialogue = match.group(2).strip()

            lines.append(DialogueLine(
                text=dialogue,
                voice_id="default",
                character_name=character,
            ))

    return DialogueScript(lines=lines)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  AXIOM VOX Multi-voice Data Models Demo")
    print("=" * 70)

    # Create a dialogue script
    script = DialogueScript(
        lines=[
            DialogueLine(
                text="Welcome to our presentation on quantum computing.",
                voice_id="announcer",
                character_name="Host",
            ),
            DialogueLine(
                text="Thank you. Let me explain the fundamentals.",
                voice_id="expert",
                character_name="Dr. Smith",
            ),
            DialogueLine(
                text="Quantum bits, or qubits, can exist in multiple states.",
                voice_id="expert",
                character_name="Dr. Smith",
            ),
            DialogueLine(
                text="That sounds fascinating! How does that work?",
                voice_id="conversational",
                character_name="Interviewer",
            ),
            DialogueLine(
                text="Through a phenomenon called superposition.",
                voice_id="expert",
                character_name="Dr. Smith",
            ),
        ],
        title="Quantum Computing Interview",
        default_transition=TransitionStyle.BREATH_PAUSE,
    )

    print(f"\nScript: {script.title}")
    print(f"  Lines: {script.line_count}")
    print(f"  Voices: {script.voices_used}")
    print(f"  Switches: {script.switch_count}")

    print("\nVoice Switches:")
    for switch in script.get_voice_switches():
        print(f"  {switch.from_voice_id or 'START'} -> {switch.to_voice_id} at line {switch.position}")

    print("\nMerged Segments:")
    segments = merge_consecutive_lines(script.lines)
    for seg in segments:
        print(f"  [{seg.voice_id}] Lines {seg.start_line_index}-{seg.end_line_index}: {seg.text[:50]}...")

    print("\nValidation:")
    issues = script.validate()
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  No issues found")

    print("\n" + "=" * 70)
