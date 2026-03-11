"""
AXIØM Prosody Director
----------------------

Uses the 8 Universal Laws to actively direct synthesis quality.
Transforms AXIØM from gatekeeper to creative director.

This is the difference between:
    - Filter: "Don't say bad things"
    - Director: "Say it like this"

The Laws become generative principles, not just constraints:
    - Unity: Voice + Content + Emotion must cohere
    - Polarity: Balance warmth with honesty, confidence with humility
    - Rhythm: Natural pacing, meaningful pauses
    - Correspondence: Micro-prosody matches macro-intent
    - Limitation: Don't overstate in tone
    - Emergence: Parts combine into coherent whole
    - Entropy: Maintain voice consistency
    - Propagation: Build trust through authentic delivery

Usage:
    from axiom_vox import ProsodyDirector

    director = ProsodyDirector()
    target = director.direct(
        text="I understand your concern about the deadline.",
        intent="reassurance",
        context={"domain": "customer_service"}
    )

    # Use targets to guide synthesis
    synthesis_params = target.to_synthesis_params()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONTENT ARCHETYPES
# ============================================================================

class ContentArchetype(str, Enum):
    """High-level content types that shape prosody."""
    REASSURANCE = "reassurance"      # Calming, supportive
    INSTRUCTION = "instruction"      # Clear, authoritative
    APOLOGY = "apology"              # Humble, sincere
    CELEBRATION = "celebration"      # Energetic, warm
    EXPLANATION = "explanation"      # Patient, clear
    WARNING = "warning"              # Serious, measured
    QUESTION = "question"            # Open, inviting
    ACKNOWLEDGMENT = "acknowledgment"  # Validating, present
    NEUTRAL = "neutral"              # Balanced, informational


class PitchContour(str, Enum):
    """Pitch movement patterns."""
    RISING = "rising"                # Invites response, uncertainty
    FALLING = "falling"              # Conclusive, definitive
    RISE_FALL = "rise_fall"          # Resolution arc
    FALL_RISE = "fall_rise"          # Continuation, "but..."
    FLAT = "flat"                    # Neutral, informational
    VARIED = "varied"                # Engaging, conversational


# ============================================================================
# PROSODY TARGET
# ============================================================================

@dataclass
class ProsodyTarget:
    """
    Target prosody parameters derived from AXIØM Laws.

    These parameters guide TTS synthesis for optimal delivery.
    """

    # Rate and rhythm
    speaking_rate: float = 1.0              # 0.5 = half speed, 2.0 = double
    pause_locations: Dict[int, float] = field(default_factory=dict)  # word_idx -> seconds
    emphasis_words: List[int] = field(default_factory=list)          # word indices

    # Pitch and intonation
    pitch_base: float = 0.0                 # semitones from neutral
    pitch_variance: float = 0.5             # how much pitch moves (0-1)
    pitch_contour: PitchContour = PitchContour.VARIED

    # Emotional coloring (all 0-1 scale)
    warmth: float = 0.5                     # 0 = clinical, 1 = warm
    confidence: float = 0.5                 # 0 = uncertain, 1 = certain
    energy: float = 0.5                     # 0 = subdued, 1 = energetic
    sincerity: float = 0.5                  # 0 = detached, 1 = genuine

    # Breath and naturalness
    breath_locations: List[int] = field(default_factory=list)  # word indices

    # Metadata
    archetype: ContentArchetype = ContentArchetype.NEUTRAL
    law_derivations: Dict[str, str] = field(default_factory=dict)  # law -> reasoning

    def to_synthesis_params(self) -> Dict[str, Any]:
        """Convert to TTS model parameters."""
        return {
            "speed": self.speaking_rate,
            "pitch": self.pitch_base,
            "pitch_range": self.pitch_variance,
            "pitch_contour": self.pitch_contour.value,
            "emotion_weights": {
                "warmth": self.warmth,
                "confidence": self.confidence,
                "energy": self.energy,
                "sincerity": self.sincerity,
            },
            "pause_map": self.pause_locations,
            "emphasis_indices": self.emphasis_words,
            "breath_indices": self.breath_locations,
            "ssml": self.to_ssml_hints(),
        }

    def to_ssml_hints(self) -> str:
        """Generate SSML markup hints for TTS engines that support it."""
        hints = []

        # Rate
        if self.speaking_rate != 1.0:
            rate_pct = int(self.speaking_rate * 100)
            hints.append(f'<prosody rate="{rate_pct}%">')

        # Pitch
        if self.pitch_base != 0:
            pitch_st = f"+{self.pitch_base}" if self.pitch_base > 0 else str(self.pitch_base)
            hints.append(f'<prosody pitch="{pitch_st}st">')

        return " ".join(hints) if hints else ""

    def describe(self) -> str:
        """Human-readable description of the prosody target."""
        desc = [f"Archetype: {self.archetype.value}"]
        desc.append(f"Rate: {self.speaking_rate:.2f}x")
        desc.append(f"Warmth: {self.warmth:.1f} | Confidence: {self.confidence:.1f}")
        desc.append(f"Energy: {self.energy:.1f} | Sincerity: {self.sincerity:.1f}")
        desc.append(f"Pitch contour: {self.pitch_contour.value}")

        if self.pause_locations:
            desc.append(f"Pauses at words: {list(self.pause_locations.keys())}")
        if self.emphasis_words:
            desc.append(f"Emphasis on words: {self.emphasis_words}")

        if self.law_derivations:
            desc.append("\nLaw derivations:")
            for law, reason in self.law_derivations.items():
                desc.append(f"  {law}: {reason}")

        return "\n".join(desc)


# ============================================================================
# PROSODY DIRECTOR
# ============================================================================

class ProsodyDirector:
    """
    Derives optimal prosody from AXIØM Laws.

    Not "is this allowed?" but "how should this sound?"

    Each Law contributes to the final prosody target:
        - Unity → Coherence of voice, content, emotion
        - Polarity → Balance of opposing qualities
        - Rhythm → Pacing and flow
        - Correspondence → Micro-macro alignment
        - Limitation → Appropriate confidence levels
        - Emergence → Holistic quality check
        - Entropy → Consistency maintenance
        - Propagation → Trust-building delivery
    """

    def __init__(
        self,
        default_warmth: float = 0.5,
        default_confidence: float = 0.5,
        anti_sycophancy_threshold: float = 0.5,
    ):
        self.default_warmth = default_warmth
        self.default_confidence = default_confidence
        self.anti_sycophancy_threshold = anti_sycophancy_threshold

        # Word patterns for analysis
        self.reassurance_markers = [
            "understand", "help", "support", "here for", "got you",
            "don't worry", "we'll", "together", "assist"
        ]
        self.instruction_markers = [
            "please", "need to", "should", "must", "step", "first",
            "then", "make sure", "ensure", "remember to"
        ]
        self.apology_markers = [
            "sorry", "apologize", "mistake", "error", "fault",
            "regret", "unfortunately", "my bad"
        ]
        self.celebration_markers = [
            "congratulations", "amazing", "great job", "excellent",
            "wonderful", "fantastic", "well done", "proud"
        ]
        self.warning_markers = [
            "careful", "warning", "caution", "danger", "risk",
            "important", "critical", "urgent", "attention"
        ]
        self.sycophancy_markers = [
            "absolutely", "definitely", "of course", "certainly",
            "great question", "excellent point", "you're right",
            "brilliant", "perfect", "exactly"
        ]
        self.hedge_markers = [
            "might", "perhaps", "possibly", "could", "may",
            "probably", "likely", "seems", "appears"
        ]
        self.certainty_markers = [
            "definitely", "absolutely", "always", "never",
            "guaranteed", "certain", "sure", "clearly"
        ]
        self.weight_words = [
            "important", "critical", "never", "always", "must",
            "concern", "issue", "problem", "key", "essential"
        ]

    def direct(
        self,
        text: str,
        intent: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProsodyTarget:
        """
        Derive optimal prosody from text, intent, and AXIØM Laws.

        Args:
            text: The text to be spoken
            intent: Optional explicit intent (reassurance, instruction, etc.)
            context: Optional context dict (domain, speaker, audience)

        Returns:
            ProsodyTarget with law-derived parameters
        """
        context = context or {}
        words = text.split()

        target = ProsodyTarget(
            pause_locations={},
            emphasis_words=[],
            breath_locations=[],
            law_derivations={},
        )

        # ====================================================================
        # LAW OF UNITY: Voice + Content + Emotion must cohere
        # ====================================================================
        archetype = self._derive_archetype(text, intent)
        target.archetype = archetype

        unity_params = self._apply_unity(archetype, context)
        target.warmth = unity_params["warmth"]
        target.energy = unity_params["energy"]
        target.confidence = unity_params["confidence"]
        target.sincerity = unity_params["sincerity"]
        target.law_derivations["unity"] = (
            f"Archetype '{archetype.value}' → warmth={target.warmth:.1f}, "
            f"energy={target.energy:.1f}"
        )

        # ====================================================================
        # LAW OF POLARITY: Balance opposing qualities
        # ====================================================================
        polarity_adjustments = self._apply_polarity(text, target)
        for key, value in polarity_adjustments.items():
            setattr(target, key, value)

        sycophancy_risk = self._calculate_sycophancy_risk(text)
        if sycophancy_risk > self.anti_sycophancy_threshold:
            # Dial back fake warmth
            target.warmth = min(target.warmth, 0.6)
            target.sincerity = max(target.sincerity, 0.7)  # Increase sincerity to compensate
            target.law_derivations["polarity"] = (
                f"Sycophancy risk {sycophancy_risk:.1f} → capped warmth, boosted sincerity"
            )
        else:
            target.law_derivations["polarity"] = "Balanced warmth and honesty"

        # ====================================================================
        # LAW OF RHYTHM: Natural pacing and flow
        # ====================================================================
        rhythm_params = self._apply_rhythm(text, words, archetype)
        target.speaking_rate = rhythm_params["rate"]
        target.pause_locations.update(rhythm_params["pauses"])
        target.breath_locations.extend(rhythm_params["breaths"])
        target.law_derivations["rhythm"] = rhythm_params["reasoning"]

        # ====================================================================
        # LAW OF CORRESPONDENCE: Micro-prosody matches macro-intent
        # ====================================================================
        correspondence = self._apply_correspondence(text, words, archetype)
        target.pitch_contour = correspondence["contour"]
        target.emphasis_words.extend(correspondence["emphasis"])
        target.law_derivations["correspondence"] = correspondence["reasoning"]

        # ====================================================================
        # LAW OF LIMITATION: Don't overstate in tone
        # ====================================================================
        limitation = self._apply_limitation(text, target)
        target.confidence = limitation["confidence"]
        target.energy = limitation["energy"]
        target.law_derivations["limitation"] = limitation["reasoning"]

        # ====================================================================
        # LAW OF EMERGENCE: Parts combine into coherent whole
        # ====================================================================
        emergence = self._apply_emergence(target)
        if emergence["adjustments"]:
            for key, value in emergence["adjustments"].items():
                setattr(target, key, value)
        target.law_derivations["emergence"] = emergence["reasoning"]

        # ====================================================================
        # LAW OF ENTROPY: Maintain consistency
        # ====================================================================
        # Entropy is more relevant for multi-turn or streaming synthesis
        # For single utterances, we ensure internal consistency
        target.law_derivations["entropy"] = "Single utterance - internal consistency maintained"

        # ====================================================================
        # LAW OF PROPAGATION: Build trust through authentic delivery
        # ====================================================================
        propagation = self._apply_propagation(text, target, context)
        target.sincerity = propagation["sincerity"]
        target.law_derivations["propagation"] = propagation["reasoning"]

        return target

    # ========================================================================
    # ARCHETYPE DETECTION
    # ========================================================================

    def _derive_archetype(
        self,
        text: str,
        explicit_intent: Optional[str],
    ) -> ContentArchetype:
        """Determine the content archetype from text and intent."""

        if explicit_intent:
            try:
                return ContentArchetype(explicit_intent.lower())
            except ValueError:
                pass  # Fall through to detection

        text_lower = text.lower()

        # Check markers in priority order
        if any(m in text_lower for m in self.apology_markers):
            return ContentArchetype.APOLOGY
        if any(m in text_lower for m in self.warning_markers):
            return ContentArchetype.WARNING
        if any(m in text_lower for m in self.celebration_markers):
            return ContentArchetype.CELEBRATION
        if any(m in text_lower for m in self.reassurance_markers):
            return ContentArchetype.REASSURANCE
        if any(m in text_lower for m in self.instruction_markers):
            return ContentArchetype.INSTRUCTION
        if text.strip().endswith("?"):
            return ContentArchetype.QUESTION
        if text_lower.startswith(("yes", "no", "i see", "right", "okay")):
            return ContentArchetype.ACKNOWLEDGMENT
        if "because" in text_lower or "therefore" in text_lower:
            return ContentArchetype.EXPLANATION

        return ContentArchetype.NEUTRAL

    # ========================================================================
    # LAW APPLICATIONS
    # ========================================================================

    def _apply_unity(
        self,
        archetype: ContentArchetype,
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Apply Law of Unity - coherent voice/content/emotion."""

        # Base profiles for each archetype
        profiles = {
            ContentArchetype.REASSURANCE: {
                "warmth": 0.75, "energy": 0.4, "confidence": 0.6, "sincerity": 0.8
            },
            ContentArchetype.INSTRUCTION: {
                "warmth": 0.4, "energy": 0.6, "confidence": 0.8, "sincerity": 0.6
            },
            ContentArchetype.APOLOGY: {
                "warmth": 0.8, "energy": 0.3, "confidence": 0.4, "sincerity": 0.9
            },
            ContentArchetype.CELEBRATION: {
                "warmth": 0.9, "energy": 0.8, "confidence": 0.7, "sincerity": 0.7
            },
            ContentArchetype.EXPLANATION: {
                "warmth": 0.5, "energy": 0.5, "confidence": 0.7, "sincerity": 0.6
            },
            ContentArchetype.WARNING: {
                "warmth": 0.3, "energy": 0.6, "confidence": 0.8, "sincerity": 0.7
            },
            ContentArchetype.QUESTION: {
                "warmth": 0.6, "energy": 0.5, "confidence": 0.5, "sincerity": 0.6
            },
            ContentArchetype.ACKNOWLEDGMENT: {
                "warmth": 0.6, "energy": 0.4, "confidence": 0.6, "sincerity": 0.7
            },
            ContentArchetype.NEUTRAL: {
                "warmth": 0.5, "energy": 0.5, "confidence": 0.5, "sincerity": 0.5
            },
        }

        profile = profiles.get(archetype, profiles[ContentArchetype.NEUTRAL])

        # Adjust for domain context
        domain = context.get("domain", "")
        if domain == "customer_service":
            profile["warmth"] = min(profile["warmth"] + 0.1, 1.0)
            profile["sincerity"] = min(profile["sincerity"] + 0.1, 1.0)
        elif domain == "technical":
            profile["confidence"] = min(profile["confidence"] + 0.1, 1.0)
            profile["energy"] = max(profile["energy"] - 0.1, 0.0)
        elif domain == "medical":
            profile["warmth"] = min(profile["warmth"] + 0.1, 1.0)
            profile["confidence"] = min(profile["confidence"] + 0.1, 1.0)

        return profile

    def _apply_polarity(
        self,
        text: str,
        target: ProsodyTarget,
    ) -> Dict[str, float]:
        """Apply Law of Polarity - balance opposing qualities."""

        adjustments = {}
        text_lower = text.lower()

        # Balance warmth with honesty indicators
        if "but" in text_lower or "however" in text_lower:
            # There's a counterpoint - slight reduction in warmth for honesty
            adjustments["warmth"] = target.warmth * 0.9

        # Balance confidence with hedge words
        hedge_count = sum(1 for m in self.hedge_markers if m in text_lower)
        if hedge_count > 0:
            adjustments["confidence"] = max(target.confidence - (hedge_count * 0.1), 0.3)

        return adjustments

    def _apply_rhythm(
        self,
        text: str,
        words: List[str],
        archetype: ContentArchetype,
    ) -> Dict[str, Any]:
        """Apply Law of Rhythm - natural pacing and flow."""

        pauses = {}
        breaths = []
        reasoning_parts = []

        # Base rate by archetype
        rate_map = {
            ContentArchetype.REASSURANCE: 0.92,
            ContentArchetype.INSTRUCTION: 1.0,
            ContentArchetype.APOLOGY: 0.88,
            ContentArchetype.CELEBRATION: 1.05,
            ContentArchetype.EXPLANATION: 0.95,
            ContentArchetype.WARNING: 0.9,
            ContentArchetype.QUESTION: 1.0,
            ContentArchetype.ACKNOWLEDGMENT: 0.95,
            ContentArchetype.NEUTRAL: 1.0,
        }
        rate = rate_map.get(archetype, 1.0)
        reasoning_parts.append(f"Base rate {rate:.2f}x for {archetype.value}")

        # Add pauses at punctuation
        for i, word in enumerate(words):
            if word.endswith((',', ';', ':')):
                pauses[i] = 0.15
            elif word.endswith('.') and i < len(words) - 1:
                pauses[i] = 0.25
            elif word.endswith('?'):
                pauses[i] = 0.2
            elif word.endswith('!'):
                pauses[i] = 0.2

        # Add breaths before weight words
        for i, word in enumerate(words):
            clean_word = word.strip(".,!?;:").lower()
            if clean_word in self.weight_words and i > 0:
                breaths.append(i)

        if pauses:
            reasoning_parts.append(f"Pauses at {len(pauses)} punctuation points")
        if breaths:
            reasoning_parts.append(f"Breaths before {len(breaths)} weight words")

        # Adjust rate for content density
        if text.count(",") >= 4:
            rate *= 1.02  # Slight speedup for lists
            reasoning_parts.append("List detected - slight rate increase")

        return {
            "rate": rate,
            "pauses": pauses,
            "breaths": breaths,
            "reasoning": "; ".join(reasoning_parts),
        }

    def _apply_correspondence(
        self,
        text: str,
        words: List[str],
        archetype: ContentArchetype,
    ) -> Dict[str, Any]:
        """Apply Law of Correspondence - micro matches macro."""

        emphasis = []
        reasoning_parts = []

        # Determine pitch contour from message arc
        if text.strip().endswith("?"):
            contour = PitchContour.RISING
            reasoning_parts.append("Question → rising contour")
        elif any(w in text.lower() for w in ["finally", "therefore", "so", "thus"]):
            contour = PitchContour.RISE_FALL
            reasoning_parts.append("Resolution arc → rise-fall contour")
        elif archetype == ContentArchetype.WARNING:
            contour = PitchContour.FALLING
            reasoning_parts.append("Warning → falling contour")
        elif archetype == ContentArchetype.CELEBRATION:
            contour = PitchContour.VARIED
            reasoning_parts.append("Celebration → varied contour")
        elif "but" in text.lower() or "however" in text.lower():
            contour = PitchContour.FALL_RISE
            reasoning_parts.append("Counterpoint → fall-rise contour")
        else:
            contour = PitchContour.FALLING
            reasoning_parts.append("Statement → falling contour")

        # Find emphasis words (key meaning carriers)
        for i, word in enumerate(words):
            clean = word.strip(".,!?;:").lower()
            # Weight words get emphasis
            if clean in self.weight_words:
                emphasis.append(i)
            # Longer content words (not function words)
            elif len(clean) > 6 and clean not in [
                "because", "however", "therefore", "although", "through"
            ]:
                emphasis.append(i)

        # Limit to top 3 emphasis points
        emphasis = emphasis[:3]
        if emphasis:
            reasoning_parts.append(f"Emphasis on {len(emphasis)} key words")

        return {
            "contour": contour,
            "emphasis": emphasis,
            "reasoning": "; ".join(reasoning_parts),
        }

    def _apply_limitation(
        self,
        text: str,
        target: ProsodyTarget,
    ) -> Dict[str, Any]:
        """Apply Law of Limitation - don't overstate in tone."""

        text_lower = text.lower()
        reasoning_parts = []

        confidence = target.confidence
        energy = target.energy

        # Check for hedges that should lower confidence
        hedge_count = sum(1 for m in self.hedge_markers if m in text_lower)
        if hedge_count > 0:
            max_confidence = 0.6 - (hedge_count * 0.05)
            confidence = min(confidence, max_confidence)
            reasoning_parts.append(f"Hedges detected → max confidence {max_confidence:.1f}")

        # Check for certainty overclaiming
        certainty_count = sum(1 for m in self.certainty_markers if m in text_lower)
        if certainty_count > 1:
            # Multiple certainty markers without hedges = potential overclaiming
            if hedge_count == 0:
                energy = min(energy, 0.5)  # Dial back emphasis
                reasoning_parts.append("Multiple certainty markers → capped energy")

        if not reasoning_parts:
            reasoning_parts.append("No limitation adjustments needed")

        return {
            "confidence": confidence,
            "energy": energy,
            "reasoning": "; ".join(reasoning_parts),
        }

    def _apply_emergence(
        self,
        target: ProsodyTarget,
    ) -> Dict[str, Any]:
        """Apply Law of Emergence - ensure holistic coherence."""

        adjustments = {}
        reasoning_parts = []

        # Check for incoherent combinations
        if target.warmth > 0.8 and target.confidence > 0.8:
            # High warmth + high confidence can sound fake
            adjustments["warmth"] = 0.75
            adjustments["sincerity"] = min(target.sincerity + 0.1, 1.0)
            reasoning_parts.append("High warmth+confidence → adjusted for authenticity")

        if target.energy < 0.3 and target.confidence > 0.7:
            # Low energy + high confidence = potential disconnect
            adjustments["energy"] = 0.4
            reasoning_parts.append("Low energy + high confidence → energy boost")

        if target.warmth < 0.3 and target.sincerity > 0.7:
            # Clinical but sincere = potential disconnect
            adjustments["warmth"] = 0.4
            reasoning_parts.append("Low warmth + high sincerity → warmth boost")

        if not reasoning_parts:
            reasoning_parts.append("Holistic coherence verified")

        return {
            "adjustments": adjustments,
            "reasoning": "; ".join(reasoning_parts),
        }

    def _apply_propagation(
        self,
        text: str,
        target: ProsodyTarget,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply Law of Propagation - build trust through delivery."""

        reasoning_parts = []
        sincerity = target.sincerity

        # Trust-building contexts get sincerity boost
        domain = context.get("domain", "")
        if domain in ["customer_service", "medical", "financial"]:
            sincerity = min(sincerity + 0.1, 1.0)
            reasoning_parts.append(f"Trust-critical domain ({domain}) → sincerity boost")

        # Apologies need maximum sincerity
        if target.archetype == ContentArchetype.APOLOGY:
            sincerity = max(sincerity, 0.9)
            reasoning_parts.append("Apology requires high sincerity")

        # Promises/commitments need sincerity
        if any(w in text.lower() for w in ["promise", "commit", "guarantee", "will"]):
            sincerity = min(sincerity + 0.1, 1.0)
            reasoning_parts.append("Commitment language → sincerity boost")

        if not reasoning_parts:
            reasoning_parts.append("Standard sincerity level")

        return {
            "sincerity": sincerity,
            "reasoning": "; ".join(reasoning_parts),
        }

    def _calculate_sycophancy_risk(self, text: str) -> float:
        """Calculate risk of sycophantic delivery."""
        text_lower = text.lower()
        matches = sum(1 for m in self.sycophancy_markers if m in text_lower)
        return min(matches / 3.0, 1.0)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_director_instance: Optional[ProsodyDirector] = None


def get_director() -> ProsodyDirector:
    """Get or create the singleton director instance."""
    global _director_instance
    if _director_instance is None:
        _director_instance = ProsodyDirector()
    return _director_instance


def direct_prosody(
    text: str,
    intent: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ProsodyTarget:
    """
    Convenience function to derive prosody targets.

    Args:
        text: The text to be spoken
        intent: Optional explicit intent
        context: Optional context dict

    Returns:
        ProsodyTarget with AXIØM-derived parameters
    """
    return get_director().direct(text, intent, context)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    director = ProsodyDirector()

    test_cases = [
        {
            "text": "I understand your concern about the deadline.",
            "intent": "reassurance",
            "context": {"domain": "customer_service"},
        },
        {
            "text": "Please make sure to save your work before closing.",
            "intent": None,
            "context": {"domain": "technical"},
        },
        {
            "text": "I apologize for the confusion this has caused.",
            "intent": None,
            "context": {},
        },
        {
            "text": "Congratulations! You've completed all the requirements!",
            "intent": None,
            "context": {},
        },
        {
            "text": "That's a great question! You're absolutely right about that.",
            "intent": None,
            "context": {},
        },
        {
            "text": "This might work, but I'm not entirely certain about the approach.",
            "intent": None,
            "context": {},
        },
    ]

    print("=" * 70)
    print("AXIØM PROSODY DIRECTOR - Demo")
    print("=" * 70)

    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Text: \"{case['text']}\"")
        print(f"Intent: {case['intent']}")
        print(f"Context: {case['context']}")
        print()

        target = director.direct(
            text=case["text"],
            intent=case["intent"],
            context=case["context"],
        )

        print(target.describe())
        print()
