"""
AXIØM Multidimensional Voice Space
----------------------------------

Maps content and voices into the same dimensional space.
Uses AXIØM's Law of Correspondence for optimal matching.

"As above, so below" → Content dimensions correspond to voice dimensions

Dimensions:
    - Formality: casual ←→ formal
    - Temperature: cold ←→ warm
    - Energy: calm ←→ urgent
    - Authority: supportive ←→ authoritative
    - Abstraction: concrete ←→ abstract
    - Intimacy: distant ←→ intimate
    - Certainty: tentative ←→ definitive
    - Complexity: simple ←→ complex

Each dimension is a spectrum (-1.0 to +1.0), not binary.
Content and voices both have coordinates in this space.
The Director finds the voice that best corresponds to the content.

Usage:
    from axiom_vox import VoiceSpaceDirector

    director = VoiceSpaceDirector()
    result = director.direct(
        text="We need to address this security issue immediately.",
        context={"domain": "technical"}
    )

    print(result["matched_voice"])  # "expert"
    print(result["inflections"])    # TTS parameters to reach content vector
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# VOICE VECTOR
# ============================================================================

@dataclass
class VoiceVector:
    """
    A point in multidimensional voice space.

    All values range from -1.0 to 1.0:
        -1.0 = full left of spectrum
         0.0 = neutral/center
        +1.0 = full right of spectrum
    """

    # Core dimensions
    formality: float = 0.0      # casual (-1) ←→ formal (+1)
    temperature: float = 0.0    # cold (-1) ←→ warm (+1)
    energy: float = 0.0         # calm (-1) ←→ urgent (+1)
    authority: float = 0.0      # supportive (-1) ←→ authoritative (+1)

    # Extended dimensions
    abstraction: float = 0.0    # concrete (-1) ←→ abstract (+1)
    intimacy: float = 0.0       # distant (-1) ←→ intimate (+1)
    certainty: float = 0.0      # tentative (-1) ←→ definitive (+1)
    complexity: float = 0.0     # simple (-1) ←→ complex (+1)

    def distance_to(self, other: 'VoiceVector') -> float:
        """Euclidean distance in voice space."""
        dims = [
            (self.formality - other.formality) ** 2,
            (self.temperature - other.temperature) ** 2,
            (self.energy - other.energy) ** 2,
            (self.authority - other.authority) ** 2,
            (self.abstraction - other.abstraction) ** 2,
            (self.intimacy - other.intimacy) ** 2,
            (self.certainty - other.certainty) ** 2,
            (self.complexity - other.complexity) ** 2,
        ]
        return math.sqrt(sum(dims))

    def blend_toward(self, target: 'VoiceVector', strength: float = 0.5) -> 'VoiceVector':
        """Create a blended vector moving toward target."""
        return VoiceVector(
            formality=self._blend(self.formality, target.formality, strength),
            temperature=self._blend(self.temperature, target.temperature, strength),
            energy=self._blend(self.energy, target.energy, strength),
            authority=self._blend(self.authority, target.authority, strength),
            abstraction=self._blend(self.abstraction, target.abstraction, strength),
            intimacy=self._blend(self.intimacy, target.intimacy, strength),
            certainty=self._blend(self.certainty, target.certainty, strength),
            complexity=self._blend(self.complexity, target.complexity, strength),
        )

    def _blend(self, a: float, b: float, strength: float) -> float:
        return a + (b - a) * strength

    def to_inflection_params(self) -> Dict[str, float]:
        """Convert vector to TTS inflection parameters."""
        return {
            # Pitch mapping
            "pitch_base": -self.authority * 2.0,  # authoritative = lower pitch
            "pitch_variance": 0.3 + (self.energy + 1) * 0.15,  # urgent = more variance

            # Rate mapping
            "speaking_rate": 1.0 + (self.energy * 0.12) - (self.complexity * 0.08),

            # Emotional mapping (normalize to 0-1)
            "warmth": (self.temperature + 1) / 2,
            "confidence": (self.certainty + 1) / 2,
            "energy": (self.energy + 1) / 2,

            # Style mapping
            "breathiness": max(0, (self.intimacy + 1) / 2 * 0.4),  # intimate = breathier
            "resonance": (self.formality + 1) / 2,  # formal = more resonant
            "precision": (self.formality + 1) / 2,  # formal = more precise articulation
        }

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "formality": self.formality,
            "temperature": self.temperature,
            "energy": self.energy,
            "authority": self.authority,
            "abstraction": self.abstraction,
            "intimacy": self.intimacy,
            "certainty": self.certainty,
            "complexity": self.complexity,
        }

    def describe(self) -> str:
        """Human-readable description of the vector."""
        parts = []

        if abs(self.formality) > 0.3:
            parts.append("formal" if self.formality > 0 else "casual")
        if abs(self.temperature) > 0.3:
            parts.append("warm" if self.temperature > 0 else "cool")
        if abs(self.energy) > 0.3:
            parts.append("energetic" if self.energy > 0 else "calm")
        if abs(self.authority) > 0.3:
            parts.append("authoritative" if self.authority > 0 else "supportive")
        if abs(self.certainty) > 0.3:
            parts.append("confident" if self.certainty > 0 else "tentative")
        if abs(self.intimacy) > 0.3:
            parts.append("intimate" if self.intimacy > 0 else "distant")

        if not parts:
            return "neutral"

        return ", ".join(parts)


# ============================================================================
# VOICE PROFILE
# ============================================================================

@dataclass
class VoiceProfile:
    """A registered voice with its dimensional profile."""

    voice_id: str
    name: str
    vector: VoiceVector

    # Voice characteristics
    gender: Optional[str] = None
    age_range: Optional[str] = None
    accent: Optional[str] = None
    description: Optional[str] = None

    # Inflection ranges (how much this voice can flex on each dimension)
    formality_range: Tuple[float, float] = (-0.5, 0.5)
    temperature_range: Tuple[float, float] = (-0.5, 0.5)
    energy_range: Tuple[float, float] = (-0.5, 0.5)
    authority_range: Tuple[float, float] = (-0.4, 0.4)

    def can_reach(self, target: VoiceVector, tolerance: float = 0.3) -> bool:
        """Check if this voice can flex to reach target vector."""
        checks = [
            self._in_range(self.vector.formality, target.formality,
                          self.formality_range, tolerance),
            self._in_range(self.vector.temperature, target.temperature,
                          self.temperature_range, tolerance),
            self._in_range(self.vector.energy, target.energy,
                          self.energy_range, tolerance),
            self._in_range(self.vector.authority, target.authority,
                          self.authority_range, tolerance),
        ]
        return all(checks)

    def _in_range(
        self,
        base: float,
        target: float,
        flex_range: Tuple[float, float],
        tolerance: float,
    ) -> bool:
        """Check if target is within base + range + tolerance."""
        low = base + flex_range[0] - tolerance
        high = base + flex_range[1] + tolerance
        return low <= target <= high

    def inflection_to_reach(self, target: VoiceVector) -> Dict[str, float]:
        """
        Calculate inflection adjustments to reach target.

        Returns delta from voice's base vector as TTS parameters.
        """
        delta = VoiceVector(
            formality=self._clamp_delta(target.formality - self.vector.formality,
                                        self.formality_range),
            temperature=self._clamp_delta(target.temperature - self.vector.temperature,
                                          self.temperature_range),
            energy=self._clamp_delta(target.energy - self.vector.energy,
                                     self.energy_range),
            authority=self._clamp_delta(target.authority - self.vector.authority,
                                        self.authority_range),
            abstraction=target.abstraction - self.vector.abstraction,
            intimacy=target.intimacy - self.vector.intimacy,
            certainty=target.certainty - self.vector.certainty,
            complexity=target.complexity - self.vector.complexity,
        )

        # Blend base with delta for final inflection params
        result_vector = self.vector.blend_toward(target, 0.7)
        return result_vector.to_inflection_params()

    def _clamp_delta(self, delta: float, range_: Tuple[float, float]) -> float:
        """Clamp delta to within allowed range."""
        return max(range_[0], min(range_[1], delta))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "vector": self.vector.to_dict(),
            "gender": self.gender,
            "age_range": self.age_range,
            "accent": self.accent,
            "description": self.description,
        }


# ============================================================================
# VOICE SPACE DIRECTOR
# ============================================================================

class VoiceSpaceDirector:
    """
    Matches content to voices using multidimensional correspondence.

    Uses AXIØM's Law of Correspondence:
        "As above, so below" → Content dimensions match voice dimensions

    The director:
        1. Analyzes content to derive its dimensional coordinates
        2. Finds the voice whose base vector is closest to content
        3. Calculates inflection adjustments for optimal correspondence
    """

    def __init__(self):
        self.voices: Dict[str, VoiceProfile] = {}
        self._init_markers()
        self._register_default_voices()

    def _init_markers(self):
        """Initialize text analysis markers."""

        # Formality markers
        self.formal_markers = [
            "therefore", "consequently", "regarding", "pursuant", "hereby",
            "accordingly", "furthermore", "nevertheless", "notwithstanding",
            "aforementioned", "hereafter", "whereas"
        ]
        self.casual_markers = [
            "hey", "gonna", "wanna", "yeah", "cool", "awesome", "stuff",
            "things", "kinda", "sorta", "like", "you know", "right"
        ]

        # Temperature markers
        self.warm_markers = [
            "thank", "appreciate", "glad", "happy", "welcome", "love",
            "wonderful", "delighted", "pleased", "grateful", "enjoy"
        ]
        self.cold_markers = [
            "must", "require", "mandate", "failure", "violation", "penalty",
            "prohibited", "forbidden", "shall not", "terminate"
        ]

        # Energy markers
        self.urgent_markers = [
            "now", "immediately", "urgent", "critical", "asap", "emergency",
            "right away", "at once", "without delay", "time-sensitive"
        ]
        self.calm_markers = [
            "when you can", "no rush", "take your time", "eventually",
            "at your convenience", "whenever", "no pressure"
        ]

        # Authority markers
        self.authoritative_markers = [
            "you must", "you need to", "it is essential", "do not",
            "you are required", "mandatory", "non-negotiable", "comply"
        ]
        self.supportive_markers = [
            "you might", "consider", "perhaps", "i suggest", "you could",
            "one option", "it may help", "feel free"
        ]

        # Certainty markers
        self.certain_markers = [
            "definitely", "certainly", "always", "never", "guaranteed",
            "absolutely", "without doubt", "unquestionably", "clearly"
        ]
        self.uncertain_markers = [
            "might", "maybe", "perhaps", "possibly", "could", "may",
            "probably", "likely", "seems", "appears", "i think"
        ]

        # Complexity markers
        self.complex_markers = [
            "however", "although", "nevertheless", "on the other hand",
            "conversely", "notwithstanding", "in contrast"
        ]
        self.simple_markers = [
            "simply", "just", "basically", "essentially", "in short"
        ]

        # Intimacy markers
        self.intimate_markers = [
            "between us", "personally", "i feel", "honestly",
            "to be frank", "just between you and me"
        ]
        self.distant_markers = [
            "the company", "the organization", "it is noted",
            "one should", "users are advised"
        ]

    def _register_default_voices(self):
        """Register built-in voice profiles."""

        # Professional narrator - balanced, authoritative
        self.register_voice(VoiceProfile(
            voice_id="professional",
            name="Professional Narrator",
            description="Clear, balanced, suitable for business and educational content",
            vector=VoiceVector(
                formality=0.5,
                temperature=0.1,
                energy=0.1,
                authority=0.4,
                certainty=0.5,
                complexity=0.2,
            ),
            formality_range=(-0.4, 0.4),
            temperature_range=(-0.3, 0.4),
            energy_range=(-0.3, 0.4),
        ))

        # Warm conversational - friendly, approachable
        self.register_voice(VoiceProfile(
            voice_id="conversational",
            name="Warm Conversational",
            description="Friendly and approachable, great for customer-facing content",
            vector=VoiceVector(
                formality=-0.3,
                temperature=0.6,
                energy=0.1,
                authority=-0.3,
                intimacy=0.4,
                certainty=0.2,
            ),
            temperature_range=(-0.2, 0.3),
            energy_range=(-0.3, 0.5),
            authority_range=(-0.3, 0.4),
        ))

        # Authoritative expert - confident, knowledgeable
        self.register_voice(VoiceProfile(
            voice_id="expert",
            name="Authoritative Expert",
            description="Confident and knowledgeable, ideal for technical content",
            vector=VoiceVector(
                formality=0.6,
                temperature=-0.1,
                energy=0.0,
                authority=0.7,
                certainty=0.7,
                complexity=0.4,
            ),
            formality_range=(-0.2, 0.3),
            authority_range=(-0.3, 0.2),
        ))

        # Supportive guide - patient, encouraging
        self.register_voice(VoiceProfile(
            voice_id="guide",
            name="Supportive Guide",
            description="Patient and encouraging, perfect for tutorials and help content",
            vector=VoiceVector(
                formality=0.0,
                temperature=0.7,
                energy=-0.1,
                authority=-0.4,
                intimacy=0.3,
                certainty=0.2,
            ),
            temperature_range=(-0.2, 0.2),
            authority_range=(-0.2, 0.5),
        ))

        # Urgent announcer - energetic, commanding
        self.register_voice(VoiceProfile(
            voice_id="announcer",
            name="Urgent Announcer",
            description="High-energy and attention-grabbing, for alerts and announcements",
            vector=VoiceVector(
                formality=0.3,
                temperature=-0.2,
                energy=0.7,
                authority=0.6,
                certainty=0.8,
            ),
            energy_range=(-0.5, 0.2),
            authority_range=(-0.2, 0.2),
        ))

        # Calm narrator - soothing, measured
        self.register_voice(VoiceProfile(
            voice_id="calm",
            name="Calm Narrator",
            description="Soothing and measured, ideal for meditation or relaxation content",
            vector=VoiceVector(
                formality=0.1,
                temperature=0.4,
                energy=-0.6,
                authority=-0.2,
                intimacy=0.5,
            ),
            energy_range=(-0.2, 0.4),
        ))

        # Casual friend - informal, relatable
        self.register_voice(VoiceProfile(
            voice_id="casual",
            name="Casual Friend",
            description="Informal and relatable, great for social media and casual content",
            vector=VoiceVector(
                formality=-0.6,
                temperature=0.5,
                energy=0.3,
                authority=-0.5,
                intimacy=0.6,
                certainty=0.0,
            ),
            formality_range=(-0.2, 0.5),
            authority_range=(-0.2, 0.6),
        ))

        # Corporate formal - polished, professional
        self.register_voice(VoiceProfile(
            voice_id="corporate",
            name="Corporate Formal",
            description="Polished and professional, for executive and legal content",
            vector=VoiceVector(
                formality=0.8,
                temperature=-0.1,
                energy=0.0,
                authority=0.5,
                certainty=0.6,
                complexity=0.3,
            ),
            formality_range=(-0.3, 0.1),
            temperature_range=(-0.2, 0.3),
        ))

    def register_voice(self, profile: VoiceProfile):
        """Register a voice profile."""
        self.voices[profile.voice_id] = profile
        logger.debug(f"Registered voice: {profile.voice_id} ({profile.name})")

    def analyze_content(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> VoiceVector:
        """
        Analyze content to derive its dimensional coordinates.

        This is where AXIØM's laws inform the analysis:
            - Correspondence: Map text features to voice dimensions
            - Unity: Ensure all dimensions cohere
        """
        context = context or {}
        text_lower = text.lower()

        # Initialize at neutral
        vector = VoiceVector()

        # === FORMALITY DIMENSION ===
        formal_count = sum(1 for m in self.formal_markers if m in text_lower)
        casual_count = sum(1 for m in self.casual_markers if m in text_lower)
        vector.formality = self._score_dimension(formal_count, casual_count, 0.25)

        # === TEMPERATURE DIMENSION ===
        warm_count = sum(1 for m in self.warm_markers if m in text_lower)
        cold_count = sum(1 for m in self.cold_markers if m in text_lower)
        vector.temperature = self._score_dimension(warm_count, cold_count, 0.2)

        # === ENERGY DIMENSION ===
        urgent_count = sum(1 for m in self.urgent_markers if m in text_lower)
        calm_count = sum(1 for m in self.calm_markers if m in text_lower)
        vector.energy = self._score_dimension(urgent_count, calm_count, 0.25)

        # Exclamation points boost energy
        exclaim_count = text.count("!")
        vector.energy += exclaim_count * 0.15
        vector.energy = max(-1.0, min(1.0, vector.energy))

        # === AUTHORITY DIMENSION ===
        auth_count = sum(1 for m in self.authoritative_markers if m in text_lower)
        supp_count = sum(1 for m in self.supportive_markers if m in text_lower)
        vector.authority = self._score_dimension(auth_count, supp_count, 0.3)

        # === CERTAINTY DIMENSION ===
        certain_count = sum(1 for m in self.certain_markers if m in text_lower)
        uncertain_count = sum(1 for m in self.uncertain_markers if m in text_lower)
        vector.certainty = self._score_dimension(certain_count, uncertain_count, 0.25)

        # === COMPLEXITY DIMENSION ===
        complex_count = sum(1 for m in self.complex_markers if m in text_lower)
        simple_count = sum(1 for m in self.simple_markers if m in text_lower)
        vector.complexity = self._score_dimension(complex_count, simple_count, 0.2)

        # Sentence length affects complexity
        sentences = text.split(".")
        avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if avg_sentence_len > 25:
            vector.complexity += 0.2
        elif avg_sentence_len < 10:
            vector.complexity -= 0.2
        vector.complexity = max(-1.0, min(1.0, vector.complexity))

        # === INTIMACY DIMENSION ===
        intimate_count = sum(1 for m in self.intimate_markers if m in text_lower)
        distant_count = sum(1 for m in self.distant_markers if m in text_lower)
        vector.intimacy = self._score_dimension(intimate_count, distant_count, 0.3)

        # First person increases intimacy
        if " i " in text_lower or text_lower.startswith("i "):
            vector.intimacy += 0.15
        # Second person increases intimacy
        if " you " in text_lower:
            vector.intimacy += 0.1
        vector.intimacy = max(-1.0, min(1.0, vector.intimacy))

        # === CONTEXT ADJUSTMENTS ===
        vector = self._apply_context_adjustments(vector, context)

        return vector

    def _score_dimension(
        self,
        positive_count: int,
        negative_count: int,
        weight: float,
    ) -> float:
        """Score a dimension based on marker counts."""
        score = (positive_count - negative_count) * weight
        return max(-1.0, min(1.0, score))

    def _apply_context_adjustments(
        self,
        vector: VoiceVector,
        context: Dict[str, Any],
    ) -> VoiceVector:
        """Apply domain-specific adjustments."""

        domain = context.get("domain", "")

        if domain == "legal":
            vector.formality = min(1.0, vector.formality + 0.3)
            vector.authority = min(1.0, vector.authority + 0.2)
            vector.certainty = min(1.0, vector.certainty + 0.2)

        elif domain == "customer_service":
            vector.temperature = min(1.0, vector.temperature + 0.25)
            vector.authority = max(-1.0, vector.authority - 0.2)
            vector.intimacy = min(1.0, vector.intimacy + 0.15)

        elif domain == "medical":
            vector.formality = min(1.0, vector.formality + 0.2)
            vector.temperature = min(1.0, vector.temperature + 0.15)
            vector.certainty = max(-1.0, vector.certainty - 0.15)  # Medical hedges

        elif domain == "technical":
            vector.formality = min(1.0, vector.formality + 0.15)
            vector.authority = min(1.0, vector.authority + 0.2)
            vector.complexity = min(1.0, vector.complexity + 0.2)

        elif domain == "casual" or domain == "social":
            vector.formality = max(-1.0, vector.formality - 0.3)
            vector.intimacy = min(1.0, vector.intimacy + 0.25)
            vector.temperature = min(1.0, vector.temperature + 0.15)

        elif domain == "emergency" or domain == "alert":
            vector.energy = min(1.0, vector.energy + 0.4)
            vector.authority = min(1.0, vector.authority + 0.3)
            vector.certainty = min(1.0, vector.certainty + 0.3)

        # Audience adjustments
        audience = context.get("audience", "")

        if audience == "children":
            vector.complexity = max(-1.0, vector.complexity - 0.4)
            vector.temperature = min(1.0, vector.temperature + 0.2)
            vector.energy = min(1.0, vector.energy + 0.15)

        elif audience == "experts":
            vector.complexity = min(1.0, vector.complexity + 0.3)
            vector.formality = min(1.0, vector.formality + 0.15)

        elif audience == "elderly":
            vector.energy = max(-1.0, vector.energy - 0.2)
            vector.temperature = min(1.0, vector.temperature + 0.15)

        return vector

    def match_voice(
        self,
        content_vector: VoiceVector,
        preferred_voice: Optional[str] = None,
        exclude_voices: Optional[List[str]] = None,
    ) -> Tuple[VoiceProfile, Dict[str, float], float]:
        """
        Find the best matching voice and calculate inflections.

        Args:
            content_vector: Target vector derived from content
            preferred_voice: Optional voice ID to use if it can reach target
            exclude_voices: Optional list of voice IDs to exclude

        Returns:
            (best_voice, inflection_params, distance)
        """
        exclude_voices = exclude_voices or []

        # If preferred voice specified and can reach target, use it
        if preferred_voice and preferred_voice in self.voices:
            if preferred_voice not in exclude_voices:
                voice = self.voices[preferred_voice]
                if voice.can_reach(content_vector):
                    inflections = voice.inflection_to_reach(content_vector)
                    distance = voice.vector.distance_to(content_vector)
                    return voice, inflections, distance

        # Find best matching voice
        best_voice = None
        best_distance = float('inf')

        # First pass: voices that can reach the target
        for voice in self.voices.values():
            if voice.voice_id in exclude_voices:
                continue
            if voice.can_reach(content_vector):
                distance = voice.vector.distance_to(content_vector)
                if distance < best_distance:
                    best_distance = distance
                    best_voice = voice

        # Second pass: if none can reach, find closest anyway
        if best_voice is None:
            for voice in self.voices.values():
                if voice.voice_id in exclude_voices:
                    continue
                distance = voice.vector.distance_to(content_vector)
                if distance < best_distance:
                    best_distance = distance
                    best_voice = voice

        if best_voice is None:
            # Fallback to professional if all excluded
            best_voice = self.voices.get("professional", list(self.voices.values())[0])
            best_distance = best_voice.vector.distance_to(content_vector)

        inflections = best_voice.inflection_to_reach(content_vector)
        return best_voice, inflections, best_distance

    def direct(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        preferred_voice: Optional[str] = None,
        exclude_voices: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full direction pipeline: analyze content → match voice → derive inflections.

        Args:
            text: The text to be spoken
            context: Optional context dict (domain, audience, etc.)
            preferred_voice: Optional voice ID to prefer
            exclude_voices: Optional list of voice IDs to exclude

        Returns:
            Complete direction result with voice, vector, and inflections
        """
        # Step 1: Analyze content
        content_vector = self.analyze_content(text, context)

        # Step 2: Match voice
        voice, inflections, distance = self.match_voice(
            content_vector,
            preferred_voice,
            exclude_voices,
        )

        # Step 3: Package result
        return {
            "text": text,
            "context": context,

            # Content analysis
            "content_vector": content_vector.to_dict(),
            "content_description": content_vector.describe(),

            # Voice matching
            "matched_voice_id": voice.voice_id,
            "matched_voice_name": voice.name,
            "voice_description": voice.description,
            "base_voice_vector": voice.vector.to_dict(),

            # Synthesis parameters
            "inflections": inflections,
            "distance": distance,
            "correspondence_score": max(0, 1.0 - distance / 3.0),  # Normalized 0-1

            # For debugging
            "vector_delta": {
                k: content_vector.to_dict()[k] - voice.vector.to_dict()[k]
                for k in content_vector.to_dict()
            },
        }

    def list_voices(self) -> List[Dict[str, Any]]:
        """List all registered voices."""
        return [v.to_dict() for v in self.voices.values()]

    def get_voice(self, voice_id: str) -> Optional[VoiceProfile]:
        """Get a voice profile by ID."""
        return self.voices.get(voice_id)


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_director_instance: Optional[VoiceSpaceDirector] = None


def get_voice_space_director() -> VoiceSpaceDirector:
    """Get or create the singleton director instance."""
    global _director_instance
    if _director_instance is None:
        _director_instance = VoiceSpaceDirector()
    return _director_instance


def direct_voice(
    text: str,
    context: Optional[Dict[str, Any]] = None,
    preferred_voice: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to direct voice matching.

    Args:
        text: The text to be spoken
        context: Optional context dict
        preferred_voice: Optional preferred voice ID

    Returns:
        Direction result with voice and inflections
    """
    return get_voice_space_director().direct(text, context, preferred_voice)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    director = VoiceSpaceDirector()

    test_cases = [
        {
            "text": "We need to address this security vulnerability immediately.",
            "context": {"domain": "technical"},
        },
        {
            "text": "Hey thanks so much for reaching out! Happy to help with that.",
            "context": {"domain": "customer_service"},
        },
        {
            "text": "Pursuant to Section 4.2 of the agreement, the licensee shall indemnify all parties.",
            "context": {"domain": "legal"},
        },
        {
            "text": "Take a deep breath. Relax your shoulders. Let go of any tension.",
            "context": {"domain": "wellness"},
        },
        {
            "text": "ALERT: System failure detected. Immediate action required!",
            "context": {"domain": "emergency"},
        },
        {
            "text": "So basically, the thing is, you just kinda click that button and boom, you're done.",
            "context": {"domain": "casual"},
        },
        {
            "text": "You might consider trying a different approach, perhaps starting with the basics.",
            "context": {},
        },
    ]

    print("=" * 80)
    print("AXIØM VOICE SPACE DIRECTOR - Multidimensional Correspondence Demo")
    print("=" * 80)

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}")
        print(f"{'='*80}")
        print(f"Text: \"{case['text'][:70]}...\"" if len(case['text']) > 70 else f"Text: \"{case['text']}\"")
        print(f"Context: {case['context']}")
        print()

        result = director.direct(case["text"], case["context"])

        print(f"Content Analysis: {result['content_description']}")
        print(f"Matched Voice: {result['matched_voice_name']} ({result['matched_voice_id']})")
        print(f"Correspondence Score: {result['correspondence_score']:.2f}")
        print()
        print("Inflection Parameters:")
        for k, v in result['inflections'].items():
            print(f"  {k}: {v:.3f}")
