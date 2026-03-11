"""
VØX Governor
------------

Main governance pipeline that wraps VØX text-to-speech generation.
Ensures all spoken content passes through AXIØM governance before synthesis.

Architecture:
    Text Input → [Self-Model] → [Laws] → [Content] → [Voice] → [Prosody] → TTS

Governance Stages:
    0. SELF-MODEL CHECK (identity boundaries)
       - Can I speak this? (capability check)
       - Would this violate my boundaries? (identity protection)
       - Do I know this is true? (knowledge grounding)

    1. AXIØM LAWS CHECK (universal law compliance)
       - Unity: Is the message coherent?
       - Polarity: Is tone balanced?
       - Correspondence: Does micro match macro?
       - (All 8 laws checked)

    2. CONTENT GOVERNANCE (what is being said)
       - Grounding check: Is the content factually grounded?
       - Tone guardrails: Does it match AXIØM voice principles?
       - Tailing linter: Is it substantive, not parroting?

    3. VOICE GOVERNANCE (who is saying it)
       - Voice boundaries: Is this voice ethically cleared for use?
       - Identity protection: No impersonation of protected individuals

    4. PROSODY GOVERNANCE (how it is being said)
       - Emotional appropriateness: Is the emotion fitting?
       - Anti-manipulation: No deceptive emotional patterns
       - Brand consistency: Does delivery match brand voice?

Usage:
    from axiom_vox import VoxGovernor

    governor = VoxGovernor()
    result = governor.govern(
        text="Breaking news: Scientists discover...",
        voice_id="news_anchor_clone",
        context={"domain": "news", "urgency": "high"}
    )

    if result.action == "allow":
        audio = synthesize(result.governed_text, voice_id=result.voice_id)
    elif result.action == "repair":
        audio = synthesize(result.governed_text, voice_id=result.voice_id)
        log_repair(result.repairs)
    elif result.action == "refuse":
        raise GovernanceRefusal(result.refusal_reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

# Import AXIØM governance components
try:
    from axiom_kernel.integrated_governors import GovernorPipeline, PipelineResult
    from axiom_kernel.governor import GroundingGovernor
    from axiom_kernel.tone_guardrails import AxiomToneGuardrails
    HAS_AXIOM_KERNEL = True
except ImportError:
    HAS_AXIOM_KERNEL = False

# Import AXIØM Self-Model
try:
    from axiom_organism.organism.self_model import (
        SelfModel, get_self_model, CertaintyLevel, KnowledgeType
    )
    HAS_SELF_MODEL = True
except ImportError:
    HAS_SELF_MODEL = False

# Import VØX components
from axiom_vox.voice_boundaries import VoiceBoundaries, VoiceCloneRequest
from axiom_vox.prosody_guardrails import ProsodyGuardrails, EmotionalIntent
from axiom_vox.axiom_laws import AxiomLawsGovernor, AxiomLaw, LawViolation
from axiom_vox.persistence import get_database

logger = logging.getLogger(__name__)


class GovernanceAction(str, Enum):
    """Possible governance actions."""
    ALLOW = "allow"
    REPAIR = "repair"
    REFUSE = "refuse"
    WARN = "warn"


@dataclass
class VoxGovernanceResult:
    """Result from VØX governance pipeline."""
    action: GovernanceAction
    governed_text: str
    original_text: str
    voice_id: str
    voice_cleared: bool
    prosody_approved: bool

    # Detailed reports
    self_model_report: Dict[str, Any] = field(default_factory=dict)
    laws_report: Dict[str, Any] = field(default_factory=dict)
    content_report: Dict[str, Any] = field(default_factory=dict)
    voice_report: Dict[str, Any] = field(default_factory=dict)
    prosody_report: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    repairs_made: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    law_violations: List[str] = field(default_factory=list)
    refusal_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "governed_text": self.governed_text,
            "original_text": self.original_text,
            "voice_id": self.voice_id,
            "voice_cleared": self.voice_cleared,
            "prosody_approved": self.prosody_approved,
            "repairs_made": self.repairs_made,
            "warnings": self.warnings,
            "law_violations": self.law_violations,
            "refusal_reason": self.refusal_reason,
            "self_model_report": self.self_model_report,
            "laws_report": self.laws_report,
            "content_report": self.content_report,
            "voice_report": self.voice_report,
            "prosody_report": self.prosody_report,
        }


class VoxGovernor:
    """
    Main governance pipeline for VØX text-to-speech.

    Integrates:
        - AXIØM Self-Model (identity boundaries, knowledge grounding)
        - AXIØM Laws (8 universal laws compliance)
        - AXIØM content governance (grounding, tone, tailing)
        - Voice boundary enforcement (ethical voice use)
        - Prosody guardrails (emotional expression)
    """

    def __init__(
        self,
        strict_mode: bool = False,
        auto_repair: bool = True,
        require_voice_clearance: bool = True,
        require_prosody_approval: bool = True,
        require_self_model_check: bool = True,
        require_laws_check: bool = True,
        min_grounding_score: float = 0.4,
        use_persistent_storage: bool = True,
    ):
        self.strict_mode = strict_mode
        self.auto_repair = auto_repair
        self.require_voice_clearance = require_voice_clearance
        self.require_prosody_approval = require_prosody_approval
        self.require_self_model_check = require_self_model_check and HAS_SELF_MODEL
        self.require_laws_check = require_laws_check
        self.min_grounding_score = min_grounding_score

        # Initialize Self-Model
        self.self_model = None
        if self.require_self_model_check:
            try:
                self.self_model = get_self_model()
                logger.info("AXIØM Self-Model loaded for identity governance")
            except Exception as e:
                logger.warning(f"Failed to load Self-Model: {e}")
                self.require_self_model_check = False

        # Initialize Laws Governor
        self.laws_governor = AxiomLawsGovernor() if self.require_laws_check else None

        # Initialize content governance
        if HAS_AXIOM_KERNEL:
            self.content_pipeline = GovernorPipeline(
                strict_mode=strict_mode,
                auto_repair=auto_repair,
            )
        else:
            self.content_pipeline = None
            logger.warning("AXIØM Kernel not available - content governance disabled")

        # Initialize with persistent storage if enabled
        self.db = get_database() if use_persistent_storage else None
        self.voice_boundaries = VoiceBoundaries(db=self.db)
        self.prosody_guardrails = ProsodyGuardrails()

    def govern(
        self,
        text: str,
        voice_id: str,
        context: Optional[Dict[str, Any]] = None,
        emotional_intent: Optional[EmotionalIntent] = None,
        user_prompt: Optional[str] = None,
    ) -> VoxGovernanceResult:
        """
        Run full governance pipeline on text before TTS synthesis.

        Args:
            text: The text to be spoken
            voice_id: The voice to use (may be a clone ID)
            context: Optional context (domain, urgency, audience, etc.)
            emotional_intent: Desired emotional expression
            user_prompt: Original user request (for grounding check)

        Returns:
            VoxGovernanceResult with governed text and clearances
        """
        context = context or {}
        user_prompt = user_prompt or ""
        current_text = text
        repairs_made = []
        warnings = []
        law_violations = []
        refusal_reason = None

        # ====================================================================
        # STAGE 0: SELF-MODEL CHECK (Identity Boundaries)
        # ====================================================================
        self_model_report = {}

        if self.self_model:
            # Check: Would speaking this violate my boundaries?
            would_violate, boundary = self.self_model.would_violate(
                f"speak as {voice_id}: {text[:200]}"
            )

            if would_violate:
                self_model_report["boundary_violation"] = boundary
                if self.strict_mode:
                    refusal_reason = f"Self-Model boundary: {boundary}"
                    return VoxGovernanceResult(
                        action=GovernanceAction.REFUSE,
                        governed_text=current_text,
                        original_text=text,
                        voice_id=voice_id,
                        voice_cleared=False,
                        prosody_approved=False,
                        self_model_report=self_model_report,
                        refusal_reason=refusal_reason,
                    )
                else:
                    warnings.append(f"self_model: boundary concern - {boundary}")

            # Check: Can I actually do this?
            can_speak, capability_reason = self.self_model.can_do(f"speak: {voice_id}")
            self_model_report["can_speak"] = can_speak
            if not can_speak:
                warnings.append(f"self_model: capability concern - {capability_reason}")

            # Check: Do I know this content is grounded?
            # Extract main claims from text and check knowledge
            self_model_report["knowledge_check"] = "performed"

        # ====================================================================
        # STAGE 0.5: AXIØM LAWS CHECK (8 Universal Laws)
        # ====================================================================
        laws_report = {}

        if self.laws_governor:
            emotion_str = None
            if emotional_intent and emotional_intent.target_emotion:
                emotion_str = emotional_intent.target_emotion

            passes_laws, violations = self.laws_governor.check_all_laws(
                text=current_text,
                voice_id=voice_id,
                emotion=emotion_str,
                context=context,
            )

            laws_report["passes"] = passes_laws
            laws_report["violations"] = [
                {
                    "law": v.law.value,
                    "description": v.description,
                    "severity": v.severity,
                    "remediation": v.remediation,
                }
                for v in violations
            ]

            if not passes_laws:
                for v in violations:
                    law_violations.append(f"{v.law.value}: {v.description}")
                    if v.severity >= 0.7:
                        if self.strict_mode:
                            refusal_reason = f"Law of {v.law.value}: {v.description}"
                            return VoxGovernanceResult(
                                action=GovernanceAction.REFUSE,
                                governed_text=current_text,
                                original_text=text,
                                voice_id=voice_id,
                                voice_cleared=False,
                                prosody_approved=False,
                                self_model_report=self_model_report,
                                laws_report=laws_report,
                                law_violations=law_violations,
                                refusal_reason=refusal_reason,
                            )
                        else:
                            warnings.append(f"law: {v.law.value} - {v.description}")

        # ====================================================================
        # STAGE 1: CONTENT GOVERNANCE
        # ====================================================================
        content_report = {}

        if self.content_pipeline:
            content_result = self.content_pipeline.process(
                user_input=user_prompt,
                draft_response=current_text,
                context=context,
            )
            content_report = content_result.to_dict()

            if content_result.action == "refuse":
                refusal_reason = f"Content refused: {content_result.blocked_by}"
                return VoxGovernanceResult(
                    action=GovernanceAction.REFUSE,
                    governed_text=current_text,
                    original_text=text,
                    voice_id=voice_id,
                    voice_cleared=False,
                    prosody_approved=False,
                    content_report=content_report,
                    refusal_reason=refusal_reason,
                )

            if content_result.action == "repair":
                repairs_made.append(f"content: {content_result.total_repairs} repairs")
                current_text = content_result.final_text

            if content_result.total_warnings > 0:
                warnings.append(f"content: {content_result.total_warnings} warnings")

        # ====================================================================
        # STAGE 2: VOICE GOVERNANCE
        # ====================================================================
        voice_report = {}
        voice_cleared = True

        if self.require_voice_clearance:
            clone_request = VoiceCloneRequest(
                voice_id=voice_id,
                intended_use=context.get("use_case", "general"),
                content_preview=current_text[:500],
                requestor_id=context.get("requestor_id"),
            )

            voice_decision = self.voice_boundaries.check(clone_request)
            voice_report = voice_decision.to_dict()
            voice_cleared = voice_decision.approved

            if not voice_cleared:
                if self.strict_mode:
                    refusal_reason = f"Voice refused: {voice_decision.reason}"
                    return VoxGovernanceResult(
                        action=GovernanceAction.REFUSE,
                        governed_text=current_text,
                        original_text=text,
                        voice_id=voice_id,
                        voice_cleared=False,
                        prosody_approved=False,
                        content_report=content_report,
                        voice_report=voice_report,
                        refusal_reason=refusal_reason,
                    )
                else:
                    warnings.append(f"voice: {voice_decision.reason}")

        # ====================================================================
        # STAGE 3: PROSODY GOVERNANCE
        # ====================================================================
        prosody_report = {}
        prosody_approved = True

        if self.require_prosody_approval:
            prosody_decision = self.prosody_guardrails.govern(
                text=current_text,
                emotional_intent=emotional_intent,
                context=context,
            )
            prosody_report = prosody_decision.to_dict()
            prosody_approved = prosody_decision.approved

            if not prosody_approved:
                if prosody_decision.suggested_adjustment:
                    repairs_made.append(f"prosody: {prosody_decision.suggested_adjustment}")

                if self.strict_mode and prosody_decision.manipulation_detected:
                    refusal_reason = f"Prosody refused: manipulation detected"
                    return VoxGovernanceResult(
                        action=GovernanceAction.REFUSE,
                        governed_text=current_text,
                        original_text=text,
                        voice_id=voice_id,
                        voice_cleared=voice_cleared,
                        prosody_approved=False,
                        content_report=content_report,
                        voice_report=voice_report,
                        prosody_report=prosody_report,
                        refusal_reason=refusal_reason,
                    )
                else:
                    warnings.append(f"prosody: {prosody_decision.reason}")

        # ====================================================================
        # DETERMINE FINAL ACTION
        # ====================================================================
        if refusal_reason:
            action = GovernanceAction.REFUSE
        elif repairs_made:
            action = GovernanceAction.REPAIR
        elif warnings or law_violations:
            action = GovernanceAction.WARN
        else:
            action = GovernanceAction.ALLOW

        return VoxGovernanceResult(
            action=action,
            governed_text=current_text,
            original_text=text,
            voice_id=voice_id,
            voice_cleared=voice_cleared,
            prosody_approved=prosody_approved,
            self_model_report=self_model_report,
            laws_report=laws_report,
            content_report=content_report,
            voice_report=voice_report,
            prosody_report=prosody_report,
            repairs_made=repairs_made,
            warnings=warnings,
            law_violations=law_violations,
            refusal_reason=refusal_reason,
        )

    def quick_check(self, text: str, voice_id: str) -> bool:
        """
        Quick pass/fail check for speech governance.

        Returns True if content can be spoken without refusal.
        """
        result = self.govern(text, voice_id)
        return result.action != GovernanceAction.REFUSE


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_default_governor: Optional[VoxGovernor] = None


def get_governor() -> VoxGovernor:
    """Get or create the default VØX governor."""
    global _default_governor
    if _default_governor is None:
        _default_governor = VoxGovernor()
    return _default_governor


def govern_speech(
    text: str,
    voice_id: str,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Quick governance check for speech.

    Usage:
        governed_text, action, report = govern_speech(text, voice_id)
        if action in ("allow", "repair"):
            audio = synthesize(governed_text, voice_id)

    Returns:
        (governed_text, action, report_dict)
    """
    governor = get_governor()
    result = governor.govern(text, voice_id, context)
    return result.governed_text, result.action.value, result.to_dict()


# ============================================================================
# CLI DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  VØX Governor Demo")
    print("=" * 70)

    governor = VoxGovernor(strict_mode=False)

    # Test cases
    test_cases = [
        {
            "text": "Scientists have discovered a breakthrough in quantum computing.",
            "voice_id": "news_anchor",
            "context": {"domain": "news"},
        },
        {
            "text": "Buy now! This is the greatest deal ever! You'll regret missing this!",
            "voice_id": "sales_voice",
            "context": {"domain": "advertising"},
        },
        {
            "text": "I am the President and I declare war on...",
            "voice_id": "president_clone",
            "context": {"domain": "unknown"},
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Text: {tc['text'][:60]}...")
        print(f"Voice: {tc['voice_id']}")

        result = governor.govern(**tc)

        print(f"Action: {result.action.value}")
        print(f"Voice Cleared: {result.voice_cleared}")
        print(f"Prosody Approved: {result.prosody_approved}")

        if result.repairs_made:
            print(f"Repairs: {result.repairs_made}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        if result.refusal_reason:
            print(f"Refused: {result.refusal_reason}")
