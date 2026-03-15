"""
VØX Custom Dictionary (Hotwords)
--------------------------------

Whisper sometimes misrecognizes domain-specific terms.
Hotwords tell faster-whisper to boost probability for specific words/phrases.

Also handles post-transcription corrections for words Whisper consistently mangles.

Storage: ~/.vox/dictionary.json

Usage:
    from axiom_vox.stt.hotwords import HotwordManager

    hw = HotwordManager()
    hw.add("AXIØM", boost=5)
    hw.add("VØX", boost=5)
    hw.add("Chatterbox", boost=3)

    # Get hotwords list for faster-whisper
    hotwords = hw.get_hotword_list()

    # Post-transcription correction
    text = hw.apply_corrections("I used axium to build vocks")
    # → "I used AXIØM to build VØX"
"""

from __future__ import annotations

import os
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

_VOX_DIR = os.path.expanduser("~/.vox")
_DICT_FILE = os.path.join(_VOX_DIR, "dictionary.json")


@dataclass
class HotwordEntry:
    """A custom dictionary entry."""
    word: str                  # The correct form
    boost: int = 3             # Whisper probability boost (1-10)
    corrections: list = field(default_factory=list)  # Common misrecognitions to auto-fix

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "boost": self.boost,
            "corrections": self.corrections,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HotwordEntry":
        return cls(
            word=d["word"],
            boost=d.get("boost", 3),
            corrections=d.get("corrections", []),
        )


class HotwordManager:
    """Manages custom dictionary for Whisper hotwords + post-transcription corrections."""

    def __init__(self, dict_path: Optional[str] = None):
        self.dict_path = dict_path or _DICT_FILE
        self.entries: Dict[str, HotwordEntry] = {}
        self._correction_map: Dict[str, str] = {}  # lowercase misspelling → correct form
        self._load()

    def _load(self):
        """Load dictionary from disk."""
        if os.path.exists(self.dict_path):
            try:
                with open(self.dict_path, "r") as f:
                    data = json.load(f)
                for entry_data in data.get("entries", []):
                    entry = HotwordEntry.from_dict(entry_data)
                    self.entries[entry.word.lower()] = entry
                self._rebuild_correction_map()
                logger.info(f"Loaded {len(self.entries)} hotwords from {self.dict_path}")
            except Exception as e:
                logger.warning(f"Failed to load dictionary: {e}")

    def _save(self):
        """Persist dictionary to disk."""
        os.makedirs(os.path.dirname(self.dict_path), exist_ok=True)
        data = {"entries": [e.to_dict() for e in self.entries.values()]}
        with open(self.dict_path, "w") as f:
            json.dump(data, f, indent=2)

    def _rebuild_correction_map(self):
        """Build lookup from misspellings to correct forms."""
        self._correction_map = {}
        for entry in self.entries.values():
            for mistake in entry.corrections:
                self._correction_map[mistake.lower()] = entry.word

    def add(self, word: str, boost: int = 3, corrections: Optional[List[str]] = None) -> HotwordEntry:
        """Add a word to the custom dictionary."""
        entry = HotwordEntry(
            word=word,
            boost=max(1, min(10, boost)),
            corrections=corrections or [],
        )
        self.entries[word.lower()] = entry
        self._rebuild_correction_map()
        self._save()
        return entry

    def remove(self, word: str) -> bool:
        """Remove a word from the dictionary."""
        key = word.lower()
        if key in self.entries:
            del self.entries[key]
            self._rebuild_correction_map()
            self._save()
            return True
        return False

    def get_hotword_list(self) -> List[str]:
        """Get list of hotword strings for faster-whisper's hotwords param."""
        return [e.word for e in self.entries.values()]

    def get_hotword_string(self) -> str:
        """Get hotwords as a single string (faster-whisper format)."""
        return " ".join(self.get_hotword_list())

    def apply_corrections(self, text: str) -> str:
        """Fix known misrecognitions in transcribed text."""
        if not self._correction_map:
            return text
        result = text
        for mistake, correct in self._correction_map.items():
            pattern = re.compile(re.escape(mistake), re.IGNORECASE)
            result = pattern.sub(correct, result)
        return result

    def list_entries(self) -> List[Dict[str, Any]]:
        """List all dictionary entries."""
        return [e.to_dict() for e in self.entries.values()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": self.list_entries(),
            "count": len(self.entries),
            "path": self.dict_path,
        }


# ============================================================================
# CONVENIENCE
# ============================================================================

_default_manager: Optional[HotwordManager] = None


def get_hotword_manager() -> HotwordManager:
    """Get or create the default hotword manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = HotwordManager()
    return _default_manager
