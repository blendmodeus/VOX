"""
VØX STT Governor
----------------

Post-transcription governance pipeline — the key differentiator
vs raw Whisper or Gladia. Applies AXIØM governance to transcribed text.

Pipeline:
    Raw Transcript → [PII Redaction] → [Content Filter] → [Laws Check]
                   → [Confidence Gate] → [Audit Log] → Governed Transcript

Usage:
    from axiom_vox.stt import STTGovernor, TranscriptionResult

    governor = STTGovernor()
    governed = governor.govern(transcription_result)

    print(governed.governed_text)  # PII redacted, filtered
    print(governed.redactions)     # ["email", "phone"]
    print(governed.flags)          # ["low_confidence_segment:2"]
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

from axiom_vox.stt.models import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)

# Try to import AXIØM governance components
try:
    from axiom_vox.axiom_laws import AxiomLawsGovernor
    HAS_LAWS_GOVERNOR = True
except ImportError:
    HAS_LAWS_GOVERNOR = False


class RedactionType(str, Enum):
    """Types of PII that can be redacted."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    ADDRESS = "address"
    NAME = "name"


class ContentFlag(str, Enum):
    """Content flags for transcribed text."""
    LOW_CONFIDENCE = "low_confidence"
    NO_SPEECH = "no_speech"
    HARMFUL_CONTENT = "harmful_content"
    SENSITIVE_TOPIC = "sensitive_topic"


@dataclass
class PIIRedaction:
    """Record of a PII redaction made."""
    type: RedactionType
    original: str
    replacement: str
    start_pos: int
    end_pos: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "replacement": self.replacement,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
        }


@dataclass
class STTGovernanceConfig:
    """Configuration for STT governance."""
    redact_pii: bool = True
    redact_emails: bool = True
    redact_phones: bool = True
    redact_ssns: bool = True
    redact_credit_cards: bool = True
    filter_content: bool = True
    check_laws: bool = True
    min_confidence: float = 0.3  # Flag segments below this
    max_no_speech_prob: float = 0.8  # Flag segments above this
    audit_log: bool = True
    pii_replacement: str = "[REDACTED]"


@dataclass
class STTGovernanceResult:
    """Result from STT governance pipeline."""
    governed_text: str
    original_text: str
    redactions: List[PIIRedaction] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    law_violations: List[str] = field(default_factory=list)
    segments_flagged: List[int] = field(default_factory=list)
    governance_applied: bool = True

    @property
    def has_redactions(self) -> bool:
        return len(self.redactions) > 0

    @property
    def has_flags(self) -> bool:
        return len(self.flags) > 0

    @property
    def is_clean(self) -> bool:
        return not self.has_redactions and not self.has_flags and not self.law_violations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "governed_text": self.governed_text,
            "original_text": self.original_text,
            "redactions": [r.to_dict() for r in self.redactions],
            "redaction_count": len(self.redactions),
            "flags": self.flags,
            "warnings": self.warnings,
            "law_violations": self.law_violations,
            "segments_flagged": self.segments_flagged,
            "is_clean": self.is_clean,
            "governance_applied": self.governance_applied,
        }


# ============================================================================
# PII PATTERNS
# ============================================================================

_PII_PATTERNS = {
    RedactionType.EMAIL: re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    ),
    RedactionType.PHONE: re.compile(
        r'(?:\+?1[-.\s]?)?'             # country code
        r'(?:\(?\d{3}\)?[-.\s]?)'        # area code
        r'\d{3}[-.\s]?\d{4}'             # number
    ),
    RedactionType.SSN: re.compile(
        r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'
    ),
    RedactionType.CREDIT_CARD: re.compile(
        r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'
    ),
}


class STTGovernor:
    """Post-transcription governance for VØX STT.

    This is what makes VØX different from raw Whisper/Gladia —
    every transcription passes through AXIØM governance.
    """

    def __init__(self, config: Optional[STTGovernanceConfig] = None):
        self.config = config or STTGovernanceConfig()
        self._laws_governor = None

        if self.config.check_laws and HAS_LAWS_GOVERNOR:
            try:
                self._laws_governor = AxiomLawsGovernor()
            except Exception as e:
                logger.warning(f"Failed to init AxiomLawsGovernor: {e}")

    def govern(
        self,
        result: TranscriptionResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> STTGovernanceResult:
        """Apply governance pipeline to a transcription result.

        Args:
            result: Raw transcription result from VoxTranscriber
            context: Optional context (user_id, domain, etc.)

        Returns:
            STTGovernanceResult with governed text and metadata
        """
        context = context or {}
        text = result.text
        redactions: List[PIIRedaction] = []
        flags: List[str] = []
        warnings: List[str] = []
        law_violations: List[str] = []
        segments_flagged: List[int] = []

        # ================================================================
        # STAGE 1: PII REDACTION
        # ================================================================
        if self.config.redact_pii:
            text, redactions = self._redact_pii(text)

        # ================================================================
        # STAGE 2: CONFIDENCE GATING
        # ================================================================
        for seg in result.segments:
            if seg.confidence < self.config.min_confidence:
                flags.append(f"low_confidence:segment_{seg.id}:{seg.confidence:.2f}")
                segments_flagged.append(seg.id)

            if seg.no_speech_prob > self.config.max_no_speech_prob:
                flags.append(f"no_speech:segment_{seg.id}:{seg.no_speech_prob:.2f}")
                segments_flagged.append(seg.id)

        # ================================================================
        # STAGE 3: CONTENT FILTERING
        # ================================================================
        if self.config.filter_content:
            content_flags = self._filter_content(text)
            flags.extend(content_flags)

        # ================================================================
        # STAGE 4: AXIØM LAWS CHECK
        # ================================================================
        if self._laws_governor and text.strip():
            try:
                passes, violations = self._laws_governor.check_all_laws(
                    text=text,
                    voice_id="stt_input",
                    context=context,
                )
                if not passes:
                    for v in violations:
                        law_violations.append(f"{v.law.value}: {v.description}")
            except Exception as e:
                warnings.append(f"laws_check_error: {e}")

        # ================================================================
        # STAGE 5: AUDIT LOG
        # ================================================================
        if self.config.audit_log:
            logger.info(
                f"STT governance: {len(redactions)} redactions, "
                f"{len(flags)} flags, {len(law_violations)} violations "
                f"(text_len={len(result.text)}, governed_len={len(text)})"
            )

        return STTGovernanceResult(
            governed_text=text,
            original_text=result.text,
            redactions=redactions,
            flags=flags,
            warnings=warnings,
            law_violations=law_violations,
            segments_flagged=list(set(segments_flagged)),
            governance_applied=True,
        )

    def _redact_pii(self, text: str) -> tuple[str, List[PIIRedaction]]:
        """Detect and redact PII from text."""
        redactions = []
        redacted = text

        type_enabled = {
            RedactionType.EMAIL: self.config.redact_emails,
            RedactionType.PHONE: self.config.redact_phones,
            RedactionType.SSN: self.config.redact_ssns,
            RedactionType.CREDIT_CARD: self.config.redact_credit_cards,
        }

        for pii_type, pattern in _PII_PATTERNS.items():
            if not type_enabled.get(pii_type, True):
                continue

            for match in pattern.finditer(text):
                redactions.append(PIIRedaction(
                    type=pii_type,
                    original=match.group(),
                    replacement=self.config.pii_replacement,
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))

        # Apply redactions in reverse order to preserve positions
        for r in sorted(redactions, key=lambda x: x.start_pos, reverse=True):
            redacted = redacted[:r.start_pos] + r.replacement + redacted[r.end_pos:]

        return redacted, redactions

    def _filter_content(self, text: str) -> List[str]:
        """Basic content filtering for transcribed text."""
        flags = []
        text_lower = text.lower()

        # Check for sensitive patterns
        sensitive_patterns = {
            "medical_info": [
                "diagnosis", "prescription", "patient",
                "medical record", "health condition",
            ],
            "financial_info": [
                "bank account", "routing number",
                "pin number", "account number",
            ],
            "legal_info": [
                "attorney-client", "privileged",
                "confidential settlement",
            ],
        }

        for category, keywords in sensitive_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    flags.append(f"sensitive:{category}:{keyword}")
                    break  # One flag per category

        return flags

    def quick_redact(self, text: str) -> str:
        """Quick PII redaction without full governance.

        Useful for lightweight processing.
        """
        redacted, _ = self._redact_pii(text)
        return redacted


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_default_governor: Optional[STTGovernor] = None


def get_stt_governor() -> STTGovernor:
    """Get or create the default STT governor."""
    global _default_governor
    if _default_governor is None:
        _default_governor = STTGovernor()
    return _default_governor


def govern_transcription(
    result: TranscriptionResult,
    context: Optional[Dict[str, Any]] = None,
) -> STTGovernanceResult:
    """Quick governance check on a transcription result."""
    return get_stt_governor().govern(result, context)
