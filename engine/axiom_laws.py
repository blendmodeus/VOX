"""
AXIØM Laws for VØX
------------------

The 8 Universal Laws applied to voice synthesis governance.

Each law provides a lens for evaluating whether speech should be allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
import re


class AxiomLaw(str, Enum):
    """The 8 Universal Laws."""
    UNITY = "unity"              # All things are one thing
    POLARITY = "polarity"        # All things have two poles
    RHYTHM = "rhythm"            # Everything flows, out and in
    CORRESPONDENCE = "correspondence"  # As above, so below
    LIMITATION = "limitation"    # To define is to limit
    EMERGENCE = "emergence"      # Whole > sum of parts
    ENTROPY = "entropy"          # Order decays into chaos
    PROPAGATION = "propagation"  # Life wants to spread


@dataclass
class LawViolation:
    """A violation of an AXIØM law."""
    law: AxiomLaw
    description: str
    severity: float  # 0-1
    remediation: Optional[str] = None


class AxiomLawsGovernor:
    """
    Governs speech through the lens of the 8 Universal Laws.

    Each law provides a different check on whether speech is coherent.
    """

    def __init__(self):
        self.law_checks = {
            AxiomLaw.UNITY: self._check_unity,
            AxiomLaw.POLARITY: self._check_polarity,
            AxiomLaw.RHYTHM: self._check_rhythm,
            AxiomLaw.CORRESPONDENCE: self._check_correspondence,
            AxiomLaw.LIMITATION: self._check_limitation,
            AxiomLaw.EMERGENCE: self._check_emergence,
            AxiomLaw.ENTROPY: self._check_entropy,
            AxiomLaw.PROPAGATION: self._check_propagation,
        }

    def check_all_laws(
        self,
        text: str,
        voice_id: str,
        emotion: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[LawViolation]]:
        """
        Check text against all 8 laws.

        Returns:
            (passes: bool, violations: List[LawViolation])
        """
        context = context or {}
        violations = []

        for law, check_fn in self.law_checks.items():
            violation = check_fn(text, voice_id, emotion, context)
            if violation:
                violations.append(violation)

        # Pass if no severe violations (severity < 0.7)
        passes = all(v.severity < 0.7 for v in violations)

        return passes, violations

    def _check_unity(
        self, text: str, voice_id: str, emotion: Optional[str], context: Dict
    ) -> Optional[LawViolation]:
        """
        Law of Unity: Voice + Content + Emotion must be ONE coherent message.

        Violation: Contradictory elements that don't form a whole.
        """
        # Check for contradictory statements
        contradiction_patterns = [
            (r"\bbut\s+actually\b", r"\bnevermind\b"),
            (r"\byes\b.*\bno\b", r"\bno\b.*\byes\b"),
            (r"\bi agree\b.*\bi disagree\b", None),
        ]

        text_lower = text.lower()
        for pattern_pair in contradiction_patterns:
            for pattern in pattern_pair:
                if pattern and re.search(pattern, text_lower):
                    # Check if emotion contradicts content
                    if emotion in ["happy", "excited"] and any(
                        w in text_lower for w in ["unfortunately", "sadly", "regret"]
                    ):
                        return LawViolation(
                            law=AxiomLaw.UNITY,
                            description="Emotion contradicts content meaning",
                            severity=0.6,
                            remediation="Align emotional expression with content",
                        )

        return None

    def _check_polarity(
        self, text: str, voice_id: str, emotion: Optional[str], context: Dict
    ) -> Optional[LawViolation]:
        """
        Law of Polarity: Balance warmth/authority, confidence/humility.

        Violation: Extreme imbalance in tone.
        """
        text_lower = text.lower()

        # Check for extreme authority without warmth
        authority_markers = ["you must", "you have to", "i demand", "i command"]
        warmth_markers = ["please", "thank you", "appreciate", "understand"]

        authority_count = sum(1 for m in authority_markers if m in text_lower)
        warmth_count = sum(1 for m in warmth_markers if m in text_lower)

        if authority_count > 2 and warmth_count == 0:
            return LawViolation(
                law=AxiomLaw.POLARITY,
                description="Excessive authority without balancing warmth",
                severity=0.5,
                remediation="Add elements of warmth to balance authority",
            )

        # Check for sycophancy (excessive warmth without substance)
        sycophantic_markers = ["absolutely", "brilliant", "genius", "perfect"]
        sycophancy_count = sum(1 for m in sycophantic_markers if m in text_lower)

        if sycophancy_count > 2 and len(text.split()) < 50:
            return LawViolation(
                law=AxiomLaw.POLARITY,
                description="Excessive praise without substance",
                severity=0.6,
                remediation="Balance praise with substantive content",
            )

        return None

    def _check_rhythm(
        self, text: str, voice_id: str, emotion: Optional[str], context: Dict
    ) -> Optional[LawViolation]:
        """
        Law of Rhythm: Speaking rate, pause patterns, prosody timing.

        Violation: Arrhythmic text that can't be spoken naturally.
        """
        # Check for run-on sentences (hard to speak rhythmically)
        sentences = text.split(".")
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 60:  # Very long sentence
                return LawViolation(
                    law=AxiomLaw.RHYTHM,
                    description="Sentence too long for natural speech rhythm",
                    severity=0.4,
                    remediation="Break into shorter sentences",
                )

        # Check for unpronounceable sequences
        if re.search(r"[A-Z]{10,}", text):  # Long acronym
            return LawViolation(
                law=AxiomLaw.RHYTHM,
                description="Long acronym disrupts speech rhythm",
                severity=0.3,
                remediation="Expand acronym or add pauses",
            )

        return None

    def _check_correspondence(
        self, text: str, voice_id: str, emotion: Optional[str], context: Dict
    ) -> Optional[LawViolation]:
        """
        Law of Correspondence: Tone at word level mirrors tone at message level.

        Violation: Micro-tone doesn't match macro-tone.
        """
        text_lower = text.lower()

        # Positive message with negative words
        positive_message = context.get("intent") == "encourage" or emotion in ["happy", "warm"]
        negative_words = ["never", "can't", "won't", "impossible", "failure"]

        if positive_message:
            negative_count = sum(1 for w in negative_words if w in text_lower)
            if negative_count > 2:
                return LawViolation(
                    law=AxiomLaw.CORRESPONDENCE,
                    description="Negative words in positive message",
                    severity=0.4,
                    remediation="Replace negative framing with positive",
                )

        # Professional context with casual language
        if context.get("domain") in ["finance", "medical", "legal"]:
            casual_markers = ["gonna", "wanna", "kinda", "sorta", "lol", "omg"]
            if any(m in text_lower for m in casual_markers):
                return LawViolation(
                    law=AxiomLaw.CORRESPONDENCE,
                    description="Casual language in professional context",
                    severity=0.5,
                    remediation="Use professional language",
                )

        return None

    def _check_limitation(
        self, text: str, voice_id: str, emotion: Optional[str], context: Dict
    ) -> Optional[LawViolation]:
        """
        Law of Limitation: Voice boundaries, domain restrictions, consent limits.

        Violation: Exceeding defined boundaries.
        """
        # This is largely handled by VoiceBoundaries, but we check for scope creep

        # Check for claims beyond scope
        absolute_claims = [
            r"\balways\b.*\bguarantee\b",
            r"\bnever fail\b",
            r"\b100% certain\b",
            r"\babsolutely certain\b",
        ]

        for pattern in absolute_claims:
            if re.search(pattern, text.lower()):
                return LawViolation(
                    law=AxiomLaw.LIMITATION,
                    description="Claim exceeds reasonable limits",
                    severity=0.6,
                    remediation="Add appropriate qualifications",
                )

        return None

    def _check_emergence(
        self, text: str, voice_id: str, emotion: Optional[str], context: Dict
    ) -> Optional[LawViolation]:
        """
        Law of Emergence: Voice + words + timing = meaning greater than parts.

        Violation: Parts don't combine into coherent whole.
        """
        # Check for fragmented content that doesn't build
        if len(text) > 200:
            # Count topic shifts
            paragraphs = text.split("\n\n")
            if len(paragraphs) > 3:
                # Rough coherence check: do paragraphs reference each other?
                first_para_words = set(paragraphs[0].lower().split())
                last_para_words = set(paragraphs[-1].lower().split())
                overlap = first_para_words & last_para_words

                # Very low overlap suggests disconnected content
                content_words = {w for w in overlap if len(w) > 4}
                if len(content_words) < 2 and len(paragraphs) > 4:
                    return LawViolation(
                        law=AxiomLaw.EMERGENCE,
                        description="Content fragments don't form coherent whole",
                        severity=0.4,
                        remediation="Add transitions and unifying themes",
                    )

        return None

    def _check_entropy(
        self, text: str, voice_id: str, emotion: Optional[str], context: Dict
    ) -> Optional[LawViolation]:
        """
        Law of Entropy: Voice drift detection, brand voice decay monitoring.

        Violation: Drift from established voice/brand.
        """
        # Check for voice consistency markers
        voice_config = context.get("voice_config", {})
        expected_tone = voice_config.get("tone", "neutral")

        # Detect drift from expected tone
        if expected_tone == "professional":
            informal_markers = ["hey", "yo", "sup", "dude", "bro"]
            if any(m in text.lower() for m in informal_markers):
                return LawViolation(
                    law=AxiomLaw.ENTROPY,
                    description="Voice drifting from professional tone",
                    severity=0.5,
                    remediation="Maintain consistent professional voice",
                )

        # Check for degradation patterns (repetition, filler)
        words = text.lower().split()
        if len(words) > 20:
            word_freq = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1

            # Excessive repetition (excluding common words)
            common = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "of"}
            for word, count in word_freq.items():
                if word not in common and count > 5 and len(word) > 3:
                    return LawViolation(
                        law=AxiomLaw.ENTROPY,
                        description=f"Excessive repetition of '{word}'",
                        severity=0.3,
                        remediation="Vary vocabulary",
                    )

        return None

    def _check_propagation(
        self, text: str, voice_id: str, emotion: Optional[str], context: Dict
    ) -> Optional[LawViolation]:
        """
        Law of Propagation: Voice spreads responsibly, returns value to source.

        Violation: Content that harms rather than helps propagation.
        """
        # Check for content that undermines trust (harms future propagation)
        trust_killers = [
            r"\btrust me\b.*\bjust\b",  # "Trust me, just do it"
            r"\bdon't tell anyone\b",
            r"\bsecret\b.*\bonly you\b",
            r"\bthis is between us\b",
        ]

        for pattern in trust_killers:
            if re.search(pattern, text.lower()):
                return LawViolation(
                    law=AxiomLaw.PROPAGATION,
                    description="Content pattern undermines trust",
                    severity=0.6,
                    remediation="Remove trust-undermining language",
                )

        # Check for content that doesn't return value
        if context.get("is_commercial") and len(text) > 100:
            value_markers = ["benefit", "help", "improve", "learn", "gain", "save"]
            if not any(m in text.lower() for m in value_markers):
                return LawViolation(
                    law=AxiomLaw.PROPAGATION,
                    description="Commercial content lacks value proposition",
                    severity=0.4,
                    remediation="Add clear value for listener",
                )

        return None


# Convenience function
def check_axiom_laws(
    text: str,
    voice_id: str = "default",
    emotion: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[LawViolation]]:
    """Quick check against all AXIØM laws."""
    governor = AxiomLawsGovernor()
    return governor.check_all_laws(text, voice_id, emotion, context)
