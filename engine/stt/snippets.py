"""
VØX Custom Snippets
-------------------

Voice-triggered text expansion. User says a trigger phrase,
VØX expands it to the configured text block.

Storage: ~/.vox/snippets.json

Usage:
    from axiom_vox.stt.snippets import SnippetManager

    sm = SnippetManager()
    sm.add("insert signature", "Best regards,\\nJeremy Brasher\\nAXIØM Labs")
    sm.add("insert date", "{date}")  # Dynamic: replaced with current date

    # Check transcribed text for triggers
    result = sm.expand("please insert signature at the end")
    # → "please Best regards,\\nJeremy Brasher\\nAXIØM Labs at the end"
"""

from __future__ import annotations

import os
import json
import re
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

_VOX_DIR = os.path.expanduser("~/.vox")
_SNIPPETS_FILE = os.path.join(_VOX_DIR, "snippets.json")


# Dynamic variables that get replaced at expansion time
_DYNAMIC_VARS = {
    "{date}": lambda: datetime.now().strftime("%B %d, %Y"),
    "{time}": lambda: datetime.now().strftime("%I:%M %p"),
    "{datetime}": lambda: datetime.now().strftime("%B %d, %Y at %I:%M %p"),
    "{today}": lambda: datetime.now().strftime("%A"),
    "{year}": lambda: str(datetime.now().year),
}


@dataclass
class Snippet:
    """A voice snippet — trigger phrase maps to expansion text."""
    trigger: str               # What the user says: "insert signature"
    expansion: str             # What gets inserted
    description: str = ""      # Optional human-readable description
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger": self.trigger,
            "expansion": self.expansion,
            "description": self.description,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Snippet":
        return cls(
            trigger=d["trigger"],
            expansion=d["expansion"],
            description=d.get("description", ""),
            enabled=d.get("enabled", True),
        )


class SnippetManager:
    """Manages voice snippets — trigger phrases that expand to text blocks."""

    def __init__(self, snippets_path: Optional[str] = None):
        self.snippets_path = snippets_path or _SNIPPETS_FILE
        self.snippets: Dict[str, Snippet] = {}
        self._load()

    def _load(self):
        """Load snippets from disk."""
        if os.path.exists(self.snippets_path):
            try:
                with open(self.snippets_path, "r") as f:
                    data = json.load(f)
                for s_data in data.get("snippets", []):
                    snippet = Snippet.from_dict(s_data)
                    self.snippets[snippet.trigger.lower()] = snippet
                logger.info(f"Loaded {len(self.snippets)} snippets from {self.snippets_path}")
            except Exception as e:
                logger.warning(f"Failed to load snippets: {e}")

    def _save(self):
        """Persist snippets to disk."""
        os.makedirs(os.path.dirname(self.snippets_path), exist_ok=True)
        data = {"snippets": [s.to_dict() for s in self.snippets.values()]}
        with open(self.snippets_path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, trigger: str, expansion: str, description: str = "") -> Snippet:
        """Add a voice snippet."""
        snippet = Snippet(
            trigger=trigger.strip(),
            expansion=expansion,
            description=description,
        )
        self.snippets[trigger.lower().strip()] = snippet
        self._save()
        return snippet

    def remove(self, trigger: str) -> bool:
        """Remove a snippet by trigger."""
        key = trigger.lower().strip()
        if key in self.snippets:
            del self.snippets[key]
            self._save()
            return True
        return False

    def expand(self, text: str) -> Tuple[str, List[str]]:
        """Check text for trigger phrases and expand them.

        Returns (expanded_text, list_of_triggers_matched).
        """
        if not self.snippets:
            return text, []

        result = text
        matched = []

        for key, snippet in self.snippets.items():
            if not snippet.enabled:
                continue

            # Case-insensitive trigger matching
            pattern = re.compile(re.escape(snippet.trigger), re.IGNORECASE)
            if pattern.search(result):
                # Resolve dynamic variables in expansion
                expansion = snippet.expansion
                for var, resolver in _DYNAMIC_VARS.items():
                    if var in expansion:
                        expansion = expansion.replace(var, resolver())

                result = pattern.sub(expansion, result)
                matched.append(snippet.trigger)

        return result, matched

    def list_snippets(self) -> List[Dict[str, Any]]:
        """List all snippets."""
        return [s.to_dict() for s in self.snippets.values()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snippets": self.list_snippets(),
            "count": len(self.snippets),
            "path": self.snippets_path,
        }


# ============================================================================
# BUILT-IN SNIPPETS (loaded on first use if no user config exists)
# ============================================================================

_BUILTINS = [
    Snippet(
        trigger="insert date",
        expansion="{date}",
        description="Current date (March 15, 2026)",
    ),
    Snippet(
        trigger="insert time",
        expansion="{time}",
        description="Current time (10:30 AM)",
    ),
    Snippet(
        trigger="insert datetime",
        expansion="{datetime}",
        description="Current date and time",
    ),
]


# ============================================================================
# CONVENIENCE
# ============================================================================

_default_manager: Optional[SnippetManager] = None


def get_snippet_manager() -> SnippetManager:
    """Get or create the default snippet manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SnippetManager()
        # Add builtins if no user snippets exist
        if not _default_manager.snippets:
            for builtin in _BUILTINS:
                _default_manager.snippets[builtin.trigger.lower()] = builtin
            _default_manager._save()
    return _default_manager
