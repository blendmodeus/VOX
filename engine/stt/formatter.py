"""
VØX Text Formatter
------------------

Post-transcription text formatting — the Glaido killer feature.
Transforms raw, messy dictation into clean, professional text.

Pipeline:
    Raw: "so um basically I was thinking we could like maybe do the thing on tuesday you know"
      ↓ [Filler Removal]  (local, no API, zero latency)
    Clean: "I was thinking we could maybe do the thing on Tuesday"
      ↓ [AI Polish]  (optional LLM pass)
    Pro: "I was thinking we could handle this on Tuesday."

Modes:
    - RAW: No formatting. What Whisper gives you.
    - CLEAN: Local filler removal + auto-punctuation. Fast, no API call.
    - PROFESSIONAL: LLM-polished output. Uses existing AI infrastructure.

Usage:
    from axiom_vox.stt.formatter import TextFormatter, FormatMode

    formatter = TextFormatter()

    # Fast local cleanup
    clean = formatter.format("so um I was like thinking about it", FormatMode.CLEAN)
    # → "I was thinking about it"

    # LLM-polished
    pro = await formatter.format_async("so um I was like thinking about it", FormatMode.PROFESSIONAL)
    # → "I was thinking about it."
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class FormatMode(str, Enum):
    """Text formatting modes."""
    RAW = "raw"              # No formatting
    CLEAN = "clean"          # Local filler removal + punctuation
    PROFESSIONAL = "professional"  # LLM-polished


# ============================================================================
# FILLER PATTERNS
# ============================================================================

# Filler words/phrases to remove — ordered by specificity (longer first)
_FILLER_PHRASES = [
    # Multi-word fillers (must match before single words)
    r"\byou know what I mean\b",
    r"\bif that makes sense\b",
    r"\bdo you know what I mean\b",
    r"\bI guess what I'm saying is\b",
    r"\bwhat I'm trying to say is\b",
    r"\bso basically\b",
    r"\bI mean like\b",
    r"\byou know\b",
    r"\bI mean\b",
    r"\bI guess\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bmore or less\b",
    r"\bor whatever\b",
    r"\bor something\b",
    r"\band stuff\b",
    r"\band things\b",
    r"\bat the end of the day\b",
    r"\bto be honest\b",
    r"\bto be fair\b",
    r"\bas a matter of fact\b",
]

# Single-word fillers
_FILLER_WORDS = [
    r"\buh+\b",
    r"\bum+\b",
    r"\buhm+\b",
    r"\bahm+\b",
    r"\bhmm+\b",
    r"\bhuh\b",
    r"\berm\b",
    r"\blike\b",     # Only when used as filler, not "I like pizza"
    r"\bbasically\b",
    r"\bliterally\b",
    r"\bactually\b",
    r"\bobviously\b",
    r"\bhonestly\b",
    r"\banyways?\b",
    r"\bright\b",    # "right, so..." filler usage
]

# Compile patterns
_FILLER_PHRASE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _FILLER_PHRASES
]

# Single-word fillers need context-aware matching to avoid false positives
# "like" is only filler in "I was like thinking" not "I like pizza"
_FILLER_WORD_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _FILLER_WORDS
]

# Context-aware filler protection
# Instead of protecting based on what comes BEFORE, check what follows.
# "like" is filler when followed by adverbs/fillers: "could like maybe"
# "like" is real when followed by nouns/objects: "I like pizza"
_LIKE_FILLER_FOLLOWERS = re.compile(
    r'\blike\s+(?:maybe|basically|really|totally|just|so|actually|'
    r'seriously|probably|definitely|honestly|literally|'
    r'\w+ing\b)',  # verb-ing: "like thinking", "like going"
    re.IGNORECASE
)

# "like" is a real word in these patterns (sounds like, looks like, I like X)
_LIKE_REAL_PATTERNS = re.compile(
    r'(?:sounds?|looks?|feels?|seems?|tastes?|smells?)\s+like\b|'
    r'(?:would|do|don\'t|didn\'t|doesn\'t|i|you|we|they)\s+like\s+(?!maybe|basically|really|totally|just|so|actually|seriously|probably|definitely|honestly|literally|\w+ing\b)',
    re.IGNORECASE
)

_FILLER_PROTECTORS = {
    "like": {"filler_followers": _LIKE_FILLER_FOLLOWERS, "protect": _LIKE_REAL_PATTERNS},
    "right": re.compile(
        r'(?:that\'s|is|was|sounds|seems|looks|got\s+it|all|just|absolutely|'
        r'exactly|the)\s+right\b',
        re.IGNORECASE
    ),
    "actually": re.compile(
        r'(?:is|was|were|are|did)\s+actually\b',
        re.IGNORECASE
    ),
}

# Repeated word detection: "I I went to the the store"
_REPEATED_WORD = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)

# False starts: "I wa- I went to the store" (word fragments)
_FALSE_START = re.compile(r'\b\w{1,3}-\s+', re.IGNORECASE)

# Multiple spaces
_MULTI_SPACE = re.compile(r'\s{2,}')

# Leading/trailing commas left by filler removal
_ORPHAN_COMMAS = re.compile(r'(?:,\s*,)|(?:^\s*,)|(?:,\s*$)')

# Sentence-starting conjunctions left hanging after filler removal
_HANGING_START = re.compile(r'^\s*(?:and|but|so|or)\s*,?\s+', re.IGNORECASE)


# ============================================================================
# AI POLISH PROMPT
# ============================================================================

_POLISH_SYSTEM_PROMPT = """You are a text formatter. Your ONLY job is to clean up dictated speech into polished, ready-to-use text.

Rules:
1. Fix grammar, punctuation, and capitalization
2. Remove any remaining filler words or verbal tics
3. Make sentences clear and professional
4. Keep the EXACT meaning and intent — do not add, remove, or change the substance
5. Do not add greetings, sign-offs, or any content the speaker didn't say
6. Do not explain what you did — output ONLY the cleaned text
7. Keep the same tone — if casual, stay casual; if formal, stay formal
8. If the input is already clean, return it unchanged

Output the cleaned text and nothing else."""


@dataclass
class FormatResult:
    """Result from text formatting."""
    formatted_text: str
    original_text: str
    mode: FormatMode
    fillers_removed: int = 0
    repeated_words_fixed: int = 0
    false_starts_fixed: int = 0
    ai_polished: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formatted_text": self.formatted_text,
            "original_text": self.original_text,
            "mode": self.mode.value,
            "fillers_removed": self.fillers_removed,
            "repeated_words_fixed": self.repeated_words_fixed,
            "false_starts_fixed": self.false_starts_fixed,
            "ai_polished": self.ai_polished,
        }


class TextFormatter:
    """Post-transcription text formatting.

    The thing that makes Glaido worth $20/mo.
    We do it locally + optionally with LLM polish.
    """

    def __init__(self):
        pass

    def format(
        self,
        text: str,
        mode: FormatMode = FormatMode.CLEAN,
    ) -> FormatResult:
        """Format text synchronously (CLEAN mode only).

        For PROFESSIONAL mode, use format_with_ai() which needs
        an async HTTP call to the LLM.
        """
        if not text or not text.strip():
            return FormatResult(
                formatted_text="",
                original_text=text,
                mode=mode,
            )

        if mode == FormatMode.RAW:
            return FormatResult(
                formatted_text=text,
                original_text=text,
                mode=mode,
            )

        # CLEAN or PROFESSIONAL: always do local cleanup first
        cleaned, stats = self._local_cleanup(text)

        return FormatResult(
            formatted_text=cleaned,
            original_text=text,
            mode=mode,
            fillers_removed=stats["fillers"],
            repeated_words_fixed=stats["repeats"],
            false_starts_fixed=stats["false_starts"],
        )

    def format_for_ai(self, text: str) -> tuple[str, str]:
        """Prepare text for AI polishing.

        Returns (locally_cleaned_text, system_prompt) so the caller
        can pass them to whatever LLM endpoint they want.
        """
        cleaned, _ = self._local_cleanup(text)
        return cleaned, _POLISH_SYSTEM_PROMPT

    def _local_cleanup(self, text: str) -> tuple[str, Dict[str, int]]:
        """Local filler removal + punctuation fix. No API calls."""
        stats = {"fillers": 0, "repeats": 0, "false_starts": 0}
        result = text

        # 1. Remove false starts ("I wa- I went")
        false_start_matches = _FALSE_START.findall(result)
        stats["false_starts"] = len(false_start_matches)
        result = _FALSE_START.sub("", result)

        # 2. Remove filler phrases (longer patterns first)
        for pattern in _FILLER_PHRASE_PATTERNS:
            matches = pattern.findall(result)
            stats["fillers"] += len(matches)
            result = pattern.sub("", result)

        # 3. Remove filler words (with context protection)
        for pattern in _FILLER_WORD_PATTERNS:
            word = pattern.pattern.replace(r'\b', '').rstrip('+').rstrip('?')

            # Check if this word has a protector
            protector = _FILLER_PROTECTORS.get(word)
            if protector and isinstance(protector, dict):
                # Dict-style protector (like "like") — check forward context
                filler_pat = protector["filler_followers"]
                protect_pat = protector["protect"]

                def _replace_like(match, _fp=filler_pat, _pp=protect_pat):
                    pos = match.start()
                    # Check if this "like" is followed by filler words → remove
                    around = result[pos:pos + 40]
                    if _fp.search(around):
                        stats["fillers"] += 1
                        return ""
                    # Check if this "like" is in a real-word pattern → keep
                    context = result[max(0, pos - 30):pos + 40]
                    if _pp.search(context):
                        return match.group()
                    # Default: remove (in dictation, "like" is usually filler)
                    stats["fillers"] += 1
                    return ""

                result = pattern.sub(_replace_like, result)

            elif protector and not isinstance(protector, dict):
                # Regex-style protector (like "right", "actually")
                protected_positions = set()
                for m in protector.finditer(result):
                    protected_positions.add(m.end() - len(word))

                def _replace_if_unprotected(match):
                    if match.start() in protected_positions:
                        return match.group()
                    stats["fillers"] += 1
                    return ""

                result = pattern.sub(_replace_if_unprotected, result)
            else:
                matches = pattern.findall(result)
                stats["fillers"] += len(matches)
                result = pattern.sub("", result)

        # 4. Fix repeated words ("I I went", "the the")
        def _fix_repeat(match):
            stats["repeats"] += 1
            return match.group(1)

        result = _REPEATED_WORD.sub(_fix_repeat, result)

        # 5. Clean up artifacts from removal
        result = _ORPHAN_COMMAS.sub("", result)
        result = _MULTI_SPACE.sub(" ", result)
        result = result.strip()

        # 6. Capitalize first letter
        if result and result[0].islower():
            result = result[0].upper() + result[1:]

        # 7. Ensure ending punctuation
        if result and result[-1] not in '.!?':
            result += '.'

        return result, stats


# ============================================================================
# CONVENIENCE
# ============================================================================

_default_formatter: Optional[TextFormatter] = None


def get_formatter() -> TextFormatter:
    """Get or create the default text formatter."""
    global _default_formatter
    if _default_formatter is None:
        _default_formatter = TextFormatter()
    return _default_formatter


def clean_text(text: str) -> str:
    """Quick clean formatting."""
    result = get_formatter().format(text, FormatMode.CLEAN)
    return result.formatted_text
