"""
VØX Config Persistence
----------------------

Save and load all VØX settings to ~/.vox/config.json.
Settings survive app restarts.

Stored settings:
    - model: Whisper model size
    - language: Transcription language
    - format: Format mode (raw/clean/professional)
    - tts: TTS enabled
    - voice: Voice profile
    - speed: Playback speed
    - wake_word: Wake word enabled
    - hotkey: Keyboard shortcut
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_VOX_DIR = os.path.expanduser("~/.vox")
_CONFIG_FILE = os.path.join(_VOX_DIR, "config.json")

# Default configuration
_DEFAULTS = {
    "model": "base",
    "language": "auto",
    "format": "clean",
    "output": "inject",
    "tts": False,
    "ai_model": "none",
    "speed": "1.0",
    "voice": "prime",
    "mode": "transcribe",
    "wake_word_enabled": False,
    "wake_words": ["vox"],
    "confidence_threshold": 0.4,
    "paragraph_gap": 1.5,
}


class VoxConfig:
    """Persistent configuration manager for VØX."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or _CONFIG_FILE
        self._config: Dict[str, Any] = dict(_DEFAULTS)
        self._load()

    def _load(self):
        """Load config from disk, merge with defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    saved = json.load(f)
                # Merge: saved values override defaults
                self._config.update(saved)
                logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

    def _save(self):
        """Persist config to disk."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a config value and persist."""
        self._config[key] = value
        self._save()

    def update(self, values: Dict[str, Any]) -> None:
        """Bulk update config values and persist."""
        self._config.update(values)
        self._save()

    def to_dict(self) -> Dict[str, Any]:
        """Get full config as dict."""
        return dict(self._config)

    def reset(self) -> None:
        """Reset to defaults."""
        self._config = dict(_DEFAULTS)
        self._save()


# ============================================================================
# CONVENIENCE
# ============================================================================

_default_config: Optional[VoxConfig] = None


def get_config() -> VoxConfig:
    """Get or create the default config manager."""
    global _default_config
    if _default_config is None:
        _default_config = VoxConfig()
    return _default_config
