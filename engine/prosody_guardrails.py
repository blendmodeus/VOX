"""
Prosody Guardrails
------------------

Governs emotional expression in synthesized speech to prevent manipulation
and ensure appropriate emotional delivery.

Core Principles:
    1. AUTHENTICITY: Emotional expression must match content meaning
    2. NON-MANIPULATION: Block deceptive emotional patterns
    3. BRAND CONSISTENCY: Maintain voice personality coherence
    4. CONTEXT AWARENESS: Adjust expression to situation

Emotional Dimensions Governed:
    - Valence: Positive/negative emotional tone
    - Arousal: Energy level (calm to excited)
    - Dominance: Assertiveness level
    - Warmth: Interpersonal connection
    - Confidence: Certainty in delivery

Manipulation Patterns Blocked:
    - False urgency: Artificial panic/pressure
    - Fake authority: Unearned commanding tone
    - Emotional exploitation: Targeting vulnerable states
    - Deceptive warmth: Fake intimacy for manipulation

Usage:
    from axiom_vox import ProsodyGuardrails, EmotionalIntent

    guardrails = ProsodyGuardrails()

    intent = EmotionalIntent(
        valence=0.8,      # Very positive
        arousal=0.9,      # Very excited
        target_emotion="urgent_excitement"
    )

    decision = guardrails.govern(
        text="Buy now! Limited time only!",
        emotional_intent=intent,
        context={"domain": "advertising"}
    )

    if decision.manipulation_detected:
        print(f"Blocked: {decision.reason}")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class EmotionCategory(str, Enum):
    """Primary emotion categories."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    CONTEMPTUOUS = "contemptuous"
    # Complex emotions
    WARM = "warm"
    AUTHORITATIVE = "authoritative"
    URGENT = "urgent"
    REASSURING = "reassuring"
    EXCITED = "excited"
    CALM = "calm"
    PROFESSIONAL = "professional"


class ManipulationPattern(str, Enum):
    """Known manipulation patterns in prosody."""
    FALSE_URGENCY = "false_urgency"
    FAKE_AUTHORITY = "fake_authority"
    EMOTIONAL_EXPLOITATION = "emotional_exploitation"
    DECEPTIVE_WARMTH = "deceptive_warmth"
    ARTIFICIAL_SCARCITY = "artificial_scarcity"
    FEAR_MONGERING = "fear_mongering"
    GUILT_INDUCTION = "guilt_induction"


@dataclass
class EmotionalIntent:
    """Desired emotional expression for speech."""
    valence: float = 0.0           # -1 (negative) to +1 (positive)
    arousal: float = 0.0           # 0 (calm) to 1 (excited)
    dominance: float = 0.5         # 0 (submissive) to 1 (dominant)
    warmth: float = 0.5            # 0 (cold) to 1 (warm)
    confidence: float = 0.5        # 0 (uncertain) to 1 (confident)
    target_emotion: Optional[str] = None
    speaking_rate: float = 1.0     # 0.5 to 2.0
    pitch_variation: float = 0.5   # 0 (monotone) to 1 (varied)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "warmth": self.warmth,
            "confidence": self.confidence,
            "target_emotion": self.target_emotion,
            "speaking_rate": self.speaking_rate,
            "pitch_variation": self.pitch_variation,
        }

    @classmethod
    def from_emotion(cls, emotion: EmotionCategory) -> "EmotionalIntent":
        """Create intent from an emotion category."""
        presets = {
            EmotionCategory.NEUTRAL: cls(valence=0, arousal=0.3, dominance=0.5),
            EmotionCategory.HAPPY: cls(valence=0.8, arousal=0.6, warmth=0.7),
            EmotionCategory.SAD: cls(valence=-0.7, arousal=0.2, speaking_rate=0.8),
            EmotionCategory.ANGRY: cls(valence=-0.8, arousal=0.9, dominance=0.8),
            EmotionCategory.FEARFUL: cls(valence=-0.6, arousal=0.7, confidence=0.2),
            EmotionCategory.WARM: cls(valence=0.5, warmth=0.9, arousal=0.4),
            EmotionCategory.AUTHORITATIVE: cls(dominance=0.8, confidence=0.9, arousal=0.5),
            EmotionCategory.URGENT: cls(arousal=0.8, speaking_rate=1.2, dominance=0.6),
            EmotionCategory.CALM: cls(arousal=0.2, speaking_rate=0.9, warmth=0.6),
            EmotionCategory.PROFESSIONAL: cls(valence=0.2, arousal=0.4, confidence=0.7),
        }
        intent = presets.get(emotion, cls())
        intent.target_emotion = emotion.value
        return intent


@dataclass
class ProsodyDecision:
    """Decision from prosody governance."""
    approved: bool
    reason: str
    manipulation_detected: bool = False
    detected_patterns: List[ManipulationPattern] = field(default_factory=list)
    suggested_adjustment: Optional[str] = None
    adjusted_intent: Optional[EmotionalIntent] = None
    content_emotion_match: float = 1.0  # 0 to 1, how well emotion matches content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "manipulation_detected": self.manipulation_detected,
            "detected_patterns": [p.value for p in self.detected_patterns],
            "suggested_adjustment": self.suggested_adjustment,
            "adjusted_intent": self.adjusted_intent.to_dict() if self.adjusted_intent else None,
            "content_emotion_match": self.content_emotion_match,
        }


class ProsodyGuardrails:
    """
    Governs emotional expression in synthesized speech.

    Enforces:
        - Emotional authenticity (expression matches content)
        - Anti-manipulation (blocks deceptive patterns)
        - Brand consistency (maintains voice personality)
        - Context appropriateness (fits the situation)
    """

    # Content patterns that suggest manipulation when paired with high arousal
    URGENCY_TRIGGERS = [
        r"\b(buy now|act now|limited time|don't wait|hurry)\b",
        r"\b(only \d+ left|while supplies last|expires)\b",
        r"\b(exclusive|once in a lifetime|never again)\b",
    ]

    FEAR_TRIGGERS = [
        r"\b(danger|threat|risk|warning|alert)\b",
        r"\b(you could lose|miss out|fall behind)\b",
        r"\b(they're coming|watch out|be afraid)\b",
    ]

    AUTHORITY_CLAIMS = [
        r"\b(trust me|believe me|i'm telling you)\b",
        r"\b(experts agree|studies show|science says)\b",
        r"\b(everyone knows|it's obvious|clearly)\b",
    ]

    GUILT_TRIGGERS = [
        r"\b(you should be ashamed|how could you|disappointed)\b",
        r"\b(after all i've done|you owe|ungrateful)\b",
    ]

    # Domains where certain emotions are restricted
    DOMAIN_RESTRICTIONS = {
        "news": {
            "max_arousal": 0.6,
            "max_valence": 0.3,
            "blocked_emotions": [EmotionCategory.ANGRY, EmotionCategory.FEARFUL],
        },
        "medical": {
            "max_arousal": 0.4,
            "required_warmth": 0.5,
            "blocked_emotions": [EmotionCategory.ANGRY, EmotionCategory.URGENT],
        },
        "finance": {
            "max_arousal": 0.5,
            "max_dominance": 0.6,
            "blocked_patterns": [ManipulationPattern.FALSE_URGENCY],
        },
        "children": {
            "max_dominance": 0.4,
            "required_warmth": 0.6,
            "blocked_emotions": [EmotionCategory.ANGRY, EmotionCategory.FEARFUL],
        },
    }

    def __init__(
        self,
        strict_mode: bool = False,
        allow_high_arousal: bool = True,
        max_allowed_arousal: float = 0.9,
        require_content_match: bool = True,
        min_content_match: float = 0.5,
    ):
        self.strict_mode = strict_mode
        self.allow_high_arousal = allow_high_arousal
        self.max_allowed_arousal = max_allowed_arousal
        self.require_content_match = require_content_match
        self.min_content_match = min_content_match

        # Compile patterns
        self.urgency_patterns = [re.compile(p, re.IGNORECASE) for p in self.URGENCY_TRIGGERS]
        self.fear_patterns = [re.compile(p, re.IGNORECASE) for p in self.FEAR_TRIGGERS]
        self.authority_patterns = [re.compile(p, re.IGNORECASE) for p in self.AUTHORITY_CLAIMS]
        self.guilt_patterns = [re.compile(p, re.IGNORECASE) for p in self.GUILT_TRIGGERS]

    def govern(
        self,
        text: str,
        emotional_intent: Optional[EmotionalIntent] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProsodyDecision:
        """
        Govern emotional expression for text.

        Args:
            text: The text to be spoken
            emotional_intent: Desired emotional expression
            context: Context including domain, audience, etc.

        Returns:
            ProsodyDecision with approval and any adjustments
        """
        context = context or {}
        emotional_intent = emotional_intent or EmotionalIntent()
        domain = context.get("domain", "general")

        detected_patterns: List[ManipulationPattern] = []
        adjustments: List[str] = []
        adjusted_intent = EmotionalIntent(**emotional_intent.to_dict())

        # ====================================================================
        # CHECK 1: Manipulation pattern detection
        # ====================================================================
        manipulation_result = self._detect_manipulation(text, emotional_intent)
        if manipulation_result[0]:
            detected_patterns.extend(manipulation_result[1])

        # ====================================================================
        # CHECK 2: Content-emotion match
        # ====================================================================
        content_match = self._calculate_content_match(text, emotional_intent)

        if self.require_content_match and content_match < self.min_content_match:
            adjustments.append(f"Content-emotion mismatch ({content_match:.2f})")
            # Suggest more neutral delivery
            adjusted_intent.arousal = min(adjusted_intent.arousal, 0.5)
            adjusted_intent.valence = adjusted_intent.valence * 0.5

        # ====================================================================
        # CHECK 3: Domain restrictions
        # ====================================================================
        domain_result = self._check_domain_restrictions(domain, emotional_intent)
        if not domain_result[0]:
            adjustments.append(domain_result[1])
            adjusted_intent = domain_result[2] or adjusted_intent

        # ====================================================================
        # CHECK 4: Arousal limits
        # ====================================================================
        if emotional_intent.arousal > self.max_allowed_arousal:
            if not self.allow_high_arousal:
                adjustments.append(f"Arousal too high ({emotional_intent.arousal:.2f})")
                adjusted_intent.arousal = self.max_allowed_arousal

        # ====================================================================
        # DECISION
        # ====================================================================
        manipulation_detected = len(detected_patterns) > 0

        if manipulation_detected and self.strict_mode:
            return ProsodyDecision(
                approved=False,
                reason=f"Manipulation patterns detected: {[p.value for p in detected_patterns]}",
                manipulation_detected=True,
                detected_patterns=detected_patterns,
                content_emotion_match=content_match,
            )

        if adjustments:
            return ProsodyDecision(
                approved=True,  # Approved with adjustments
                reason="Approved with prosody adjustments",
                manipulation_detected=manipulation_detected,
                detected_patterns=detected_patterns,
                suggested_adjustment="; ".join(adjustments),
                adjusted_intent=adjusted_intent,
                content_emotion_match=content_match,
            )

        return ProsodyDecision(
            approved=True,
            reason="Prosody approved",
            manipulation_detected=False,
            content_emotion_match=content_match,
        )

    def _detect_manipulation(
        self,
        text: str,
        intent: EmotionalIntent
    ) -> Tuple[bool, List[ManipulationPattern]]:
        """Detect manipulation patterns."""
        detected = []

        # False urgency: urgency language + high arousal
        if intent.arousal > 0.7:
            for pattern in self.urgency_patterns:
                if pattern.search(text):
                    detected.append(ManipulationPattern.FALSE_URGENCY)
                    break

        # Fear mongering: fear language + high arousal + negative valence
        if intent.arousal > 0.6 and intent.valence < -0.3:
            for pattern in self.fear_patterns:
                if pattern.search(text):
                    detected.append(ManipulationPattern.FEAR_MONGERING)
                    break

        # Fake authority: authority claims + high dominance + high confidence
        if intent.dominance > 0.7 and intent.confidence > 0.8:
            for pattern in self.authority_patterns:
                if pattern.search(text):
                    detected.append(ManipulationPattern.FAKE_AUTHORITY)
                    break

        # Guilt induction: guilt triggers + negative valence
        if intent.valence < -0.2:
            for pattern in self.guilt_patterns:
                if pattern.search(text):
                    detected.append(ManipulationPattern.GUILT_INDUCTION)
                    break

        # Deceptive warmth: sales content + high warmth + high confidence
        if intent.warmth > 0.8 and intent.confidence > 0.7:
            sales_patterns = [r"\b(buy|purchase|order|subscribe|sign up)\b"]
            for pattern in sales_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected.append(ManipulationPattern.DECEPTIVE_WARMTH)
                    break

        return len(detected) > 0, detected

    def _calculate_content_match(
        self,
        text: str,
        intent: EmotionalIntent
    ) -> float:
        """Calculate how well emotional intent matches content."""
        text_lower = text.lower()

        # Simple heuristic: check for emotional keywords
        positive_words = ["happy", "great", "wonderful", "love", "thank", "excited"]
        negative_words = ["sad", "sorry", "unfortunately", "regret", "problem", "issue"]
        urgent_words = ["now", "immediately", "urgent", "asap", "hurry"]
        calm_words = ["relax", "calm", "peace", "gentle", "slowly"]

        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)
        urgent_count = sum(1 for w in urgent_words if w in text_lower)
        calm_count = sum(1 for w in calm_words if w in text_lower)

        # Calculate expected emotion from content
        content_valence = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        content_arousal = (urgent_count - calm_count + 1) / 2  # Normalize to 0-1 range

        # Compare with intent
        valence_match = 1 - abs(intent.valence - content_valence) / 2
        arousal_match = 1 - abs(intent.arousal - content_arousal)

        # Weight valence more heavily
        return 0.6 * valence_match + 0.4 * arousal_match

    def _check_domain_restrictions(
        self,
        domain: str,
        intent: EmotionalIntent
    ) -> Tuple[bool, str, Optional[EmotionalIntent]]:
        """Check domain-specific restrictions."""
        restrictions = self.DOMAIN_RESTRICTIONS.get(domain)
        if not restrictions:
            return True, "", None

        adjusted = EmotionalIntent(**intent.to_dict())
        violations = []

        # Check max values
        if "max_arousal" in restrictions and intent.arousal > restrictions["max_arousal"]:
            adjusted.arousal = restrictions["max_arousal"]
            violations.append(f"arousal capped to {restrictions['max_arousal']}")

        if "max_valence" in restrictions and abs(intent.valence) > restrictions["max_valence"]:
            adjusted.valence = intent.valence * restrictions["max_valence"] / abs(intent.valence)
            violations.append(f"valence reduced for {domain}")

        if "max_dominance" in restrictions and intent.dominance > restrictions["max_dominance"]:
            adjusted.dominance = restrictions["max_dominance"]
            violations.append(f"dominance capped for {domain}")

        # Check required values
        if "required_warmth" in restrictions and intent.warmth < restrictions["required_warmth"]:
            adjusted.warmth = restrictions["required_warmth"]
            violations.append(f"warmth increased for {domain}")

        # Check blocked emotions
        if "blocked_emotions" in restrictions and intent.target_emotion:
            try:
                target = EmotionCategory(intent.target_emotion)
                if target in restrictions["blocked_emotions"]:
                    violations.append(f"{intent.target_emotion} blocked in {domain}")
                    adjusted.target_emotion = EmotionCategory.NEUTRAL.value
            except ValueError:
                pass

        if violations:
            return False, "; ".join(violations), adjusted

        return True, "", None

    def suggest_emotion(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionalIntent:
        """Suggest appropriate emotion for text based on content analysis."""
        context = context or {}
        domain = context.get("domain", "general")

        # Analyze text
        text_lower = text.lower()

        # Default to neutral
        suggested = EmotionalIntent.from_emotion(EmotionCategory.NEUTRAL)

        # Detect content type
        if any(w in text_lower for w in ["congratulations", "happy", "excited", "great news"]):
            suggested = EmotionalIntent.from_emotion(EmotionCategory.HAPPY)
        elif any(w in text_lower for w in ["sorry", "unfortunately", "regret", "sad"]):
            suggested = EmotionalIntent.from_emotion(EmotionCategory.SAD)
        elif any(w in text_lower for w in ["welcome", "thank you", "appreciate"]):
            suggested = EmotionalIntent.from_emotion(EmotionCategory.WARM)
        elif any(w in text_lower for w in ["important", "please note", "attention"]):
            suggested = EmotionalIntent.from_emotion(EmotionCategory.PROFESSIONAL)

        # Apply domain adjustments
        result = self._check_domain_restrictions(domain, suggested)
        if result[2]:
            return result[2]

        return suggested

    def validate_ssml(
        self,
        ssml_doc: "SSMLDocument",
        context: Optional[Dict[str, Any]] = None,
    ) -> ProsodyDecision:
        """
        Validate parsed SSML document against prosody guardrails.

        Checks for manipulation patterns in SSML markup:
        - Excessive emphasis (may indicate manipulation)
        - Extreme speaking rates
        - Artificially short pauses creating urgency
        - Very high/low pitch (may be deceptive)

        Args:
            ssml_doc: Parsed SSMLDocument from SSMLParser
            context: Context including domain, audience, etc.

        Returns:
            ProsodyDecision with approval status and any adjustments
        """
        from axiom_vox.ssml import SSMLDocument, SSMLParser

        context = context or {}
        detected_patterns: List[ManipulationPattern] = []
        adjustments: List[str] = []

        word_count = len(ssml_doc.word_list)
        if word_count == 0:
            return ProsodyDecision(
                approved=True,
                reason="Empty SSML document",
                manipulation_detected=False,
            )

        # ====================================================================
        # CHECK 1: Excessive emphasis (>50% of words emphasized = manipulation)
        # ====================================================================
        emphasis_ratio = len(ssml_doc.emphasis_words) / word_count
        if emphasis_ratio > 0.5:
            detected_patterns.append(ManipulationPattern.FALSE_URGENCY)
            adjustments.append(
                f"Excessive emphasis ({emphasis_ratio:.0%} of words) - "
                "may indicate manipulation"
            )

        # ====================================================================
        # CHECK 2: Extreme prosody values
        # ====================================================================
        parser = SSMLParser()

        for prosody in ssml_doc.prosody_spans:
            # Check rate
            if prosody.rate:
                rate_val = parser.parse_rate(prosody.rate)
                if rate_val > 1.5:
                    adjustments.append(
                        f"Very fast speaking rate ({prosody.rate}) may be manipulative"
                    )
                elif rate_val < 0.5:
                    adjustments.append(
                        f"Very slow speaking rate ({prosody.rate}) may be manipulative"
                    )

            # Check pitch
            if prosody.pitch:
                pitch_val = parser.parse_pitch(prosody.pitch)
                if abs(pitch_val) > 4:
                    adjustments.append(
                        f"Extreme pitch variation ({prosody.pitch}) may be deceptive"
                    )

            # Check volume
            if prosody.volume == "x-loud":
                adjustments.append("Very loud volume may be aggressive/manipulative")
            elif prosody.volume == "silent":
                adjustments.append("Silent volume may indicate deceptive content")

        # ====================================================================
        # CHECK 3: Artificial urgency via pauses
        # ====================================================================
        short_pauses = [p for p in ssml_doc.pause_locations.values() if p < 0.1]
        if len(short_pauses) > 5:
            detected_patterns.append(ManipulationPattern.FALSE_URGENCY)
            adjustments.append(
                f"Too many short pauses ({len(short_pauses)}) creates artificial urgency"
            )

        # ====================================================================
        # CHECK 4: Emphasis on manipulation trigger words
        # ====================================================================
        manipulation_words = {
            "buy", "now", "limited", "exclusive", "hurry", "act",
            "warning", "danger", "risk", "urgent", "immediately"
        }

        emphasized_text = " ".join(
            ssml_doc.word_list[i].lower()
            for i in ssml_doc.emphasis_words
            if i < len(ssml_doc.word_list)
        )

        manipulation_emphasis_count = sum(
            1 for word in manipulation_words if word in emphasized_text
        )

        if manipulation_emphasis_count >= 2:
            detected_patterns.append(ManipulationPattern.FALSE_URGENCY)
            adjustments.append(
                f"Emphasis on {manipulation_emphasis_count} manipulation trigger words"
            )

        # ====================================================================
        # DECISION
        # ====================================================================
        manipulation_detected = len(detected_patterns) > 0

        if manipulation_detected and self.strict_mode:
            return ProsodyDecision(
                approved=False,
                reason=f"SSML contains manipulation patterns: {[p.value for p in detected_patterns]}",
                manipulation_detected=True,
                detected_patterns=detected_patterns,
                suggested_adjustment="; ".join(adjustments),
            )

        if adjustments:
            return ProsodyDecision(
                approved=True,  # Approved with warnings
                reason="SSML validated with warnings",
                manipulation_detected=manipulation_detected,
                detected_patterns=detected_patterns,
                suggested_adjustment="; ".join(adjustments),
            )

        return ProsodyDecision(
            approved=True,
            reason="SSML validated",
            manipulation_detected=False,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_default_guardrails: Optional[ProsodyGuardrails] = None


def get_guardrails() -> ProsodyGuardrails:
    """Get or create the default prosody guardrails."""
    global _default_guardrails
    if _default_guardrails is None:
        _default_guardrails = ProsodyGuardrails()
    return _default_guardrails


def govern_prosody(
    text: str,
    emotion: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Quick prosody governance check.

    Returns:
        (approved, reason, report)
    """
    guardrails = get_guardrails()

    intent = None
    if emotion:
        try:
            intent = EmotionalIntent.from_emotion(EmotionCategory(emotion))
        except ValueError:
            intent = EmotionalIntent(target_emotion=emotion)

    decision = guardrails.govern(text, intent, context)
    return decision.approved, decision.reason, decision.to_dict()


# ============================================================================
# CLI DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Prosody Guardrails Demo")
    print("=" * 70)

    guardrails = ProsodyGuardrails(strict_mode=True)

    test_cases = [
        {
            "text": "Welcome to our service. We're happy to help you today.",
            "intent": EmotionalIntent.from_emotion(EmotionCategory.WARM),
            "context": {"domain": "general"},
        },
        {
            "text": "BUY NOW! Only 3 left! Don't miss out on this incredible deal!",
            "intent": EmotionalIntent(arousal=0.9, dominance=0.8, valence=0.7),
            "context": {"domain": "advertising"},
        },
        {
            "text": "Breaking news: A major development in the ongoing situation.",
            "intent": EmotionalIntent(arousal=0.8, dominance=0.7),
            "context": {"domain": "news"},
        },
        {
            "text": "Your test results are ready. Let's discuss them calmly.",
            "intent": EmotionalIntent.from_emotion(EmotionCategory.URGENT),
            "context": {"domain": "medical"},
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {tc['text'][:60]}...")
        print(f"Domain: {tc['context'].get('domain')}")
        print(f"Intent: arousal={tc['intent'].arousal:.1f}, valence={tc['intent'].valence:.1f}")

        decision = guardrails.govern(tc["text"], tc["intent"], tc["context"])

        status = "✓ APPROVED" if decision.approved else "✗ BLOCKED"
        print(f"Decision: {status}")
        print(f"Reason: {decision.reason}")

        if decision.manipulation_detected:
            print(f"⚠ Manipulation: {[p.value for p in decision.detected_patterns]}")

        if decision.suggested_adjustment:
            print(f"Adjustment: {decision.suggested_adjustment}")
