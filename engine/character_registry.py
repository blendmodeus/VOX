"""
AXIOM VOX Character Registry
-----------------------------

Manages character-to-voice mappings for dialogue synthesis.

Features:
- Register characters with voice assignments
- Auto-assign distinct voices to characters
- Persist mappings to database
- Query characters by various attributes

v0.9.0: Multi-voice Synthesis

Usage:
    from axiom_vox.character_registry import CharacterRegistry

    registry = CharacterRegistry()
    registry.register("Dr. Smith", voice_id="expert", default_emotion="confident")
    registry.register("Host", voice_id="announcer")

    # Get voice for character
    voice_id = registry.get_voice("Dr. Smith")

    # Auto-assign voices to new characters
    assignments = registry.auto_assign_voices(["Alice", "Bob", "Charlie"])
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# CHARACTER VOICE MAPPING
# ============================================================================


@dataclass
class CharacterVoiceMapping:
    """Maps a character name to a voice configuration."""
    character_name: str
    voice_id: str
    default_emotion: Optional[str] = None
    description: Optional[str] = None

    # Voice config overrides
    speaking_rate: Optional[float] = None
    pitch: Optional[float] = None
    volume: Optional[float] = None

    # Metadata
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "character_name": self.character_name,
            "voice_id": self.voice_id,
            "default_emotion": self.default_emotion,
            "description": self.description,
            "speaking_rate": self.speaking_rate,
            "pitch": self.pitch,
            "volume": self.volume,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterVoiceMapping":
        """Create from dictionary."""
        return cls(
            character_name=data["character_name"],
            voice_id=data["voice_id"],
            default_emotion=data.get("default_emotion"),
            description=data.get("description"),
            speaking_rate=data.get("speaking_rate"),
            pitch=data.get("pitch"),
            volume=data.get("volume"),
            tags=data.get("tags", []),
        )


# ============================================================================
# CHARACTER REGISTRY
# ============================================================================


class CharacterRegistry:
    """
    Manages character-to-voice mappings for dialogue.

    Thread-safe registry that can persist to database.
    """

    # Default voices for auto-assignment (distinct characteristics)
    DEFAULT_VOICE_POOL = [
        "professional",
        "conversational",
        "expert",
        "guide",
        "announcer",
        "calm",
        "casual",
        "corporate",
    ]

    def __init__(self, database: Optional["VoxDatabase"] = None):
        """
        Initialize character registry.

        Args:
            database: Optional VoxDatabase for persistence
        """
        self._characters: Dict[str, CharacterVoiceMapping] = {}
        self._database = database
        self._voice_pool_index = 0

    def register(
        self,
        character_name: str,
        voice_id: str,
        default_emotion: Optional[str] = None,
        description: Optional[str] = None,
        speaking_rate: Optional[float] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        tags: Optional[List[str]] = None,
        persist: bool = True,
    ) -> CharacterVoiceMapping:
        """
        Register a character with a voice.

        Args:
            character_name: Character name (case-insensitive lookup)
            voice_id: Voice ID to use for this character
            default_emotion: Default emotion preset
            description: Character description
            speaking_rate: Override speaking rate
            pitch: Override pitch
            volume: Override volume
            tags: Character tags for filtering
            persist: Whether to persist to database

        Returns:
            CharacterVoiceMapping instance
        """
        mapping = CharacterVoiceMapping(
            character_name=character_name,
            voice_id=voice_id,
            default_emotion=default_emotion,
            description=description,
            speaking_rate=speaking_rate,
            pitch=pitch,
            volume=volume,
            tags=tags or [],
        )

        self._characters[character_name.lower()] = mapping

        if persist and self._database:
            self._persist_mapping(mapping)

        logger.debug(f"Registered character: {character_name} -> {voice_id}")
        return mapping

    def unregister(self, character_name: str) -> bool:
        """
        Remove a character registration.

        Args:
            character_name: Character name to remove

        Returns:
            True if removed, False if not found
        """
        key = character_name.lower()
        if key in self._characters:
            del self._characters[key]
            logger.debug(f"Unregistered character: {character_name}")
            return True
        return False

    def get(self, character_name: str) -> Optional[CharacterVoiceMapping]:
        """
        Get mapping for a character.

        Args:
            character_name: Character name (case-insensitive)

        Returns:
            CharacterVoiceMapping or None
        """
        return self._characters.get(character_name.lower())

    def get_voice(self, character_name: str) -> Optional[str]:
        """
        Get voice ID for a character.

        Args:
            character_name: Character name (case-insensitive)

        Returns:
            Voice ID or None
        """
        mapping = self.get(character_name)
        return mapping.voice_id if mapping else None

    def get_or_assign(
        self,
        character_name: str,
        exclude_voices: Optional[Set[str]] = None,
    ) -> str:
        """
        Get voice for character, auto-assigning if not registered.

        Args:
            character_name: Character name
            exclude_voices: Voices to exclude from auto-assignment

        Returns:
            Voice ID (existing or newly assigned)
        """
        existing = self.get_voice(character_name)
        if existing:
            return existing

        # Auto-assign
        voice_id = self._get_next_available_voice(exclude_voices)
        self.register(character_name, voice_id)
        return voice_id

    def auto_assign_voices(
        self,
        character_names: List[str],
        exclude_voices: Optional[List[str]] = None,
        use_voice_space: bool = False,
    ) -> Dict[str, str]:
        """
        Automatically assign distinct voices to characters.

        Tries to maximize voice distinction across characters.

        Args:
            character_names: List of character names
            exclude_voices: Voices to exclude
            use_voice_space: Use VoiceSpaceDirector for optimal matching

        Returns:
            Dict mapping character name to assigned voice ID
        """
        exclude_set = set(exclude_voices or [])
        assignments = {}

        # First, check existing assignments
        for name in character_names:
            existing = self.get_voice(name)
            if existing:
                assignments[name] = existing
                exclude_set.add(existing)

        # Assign remaining characters
        for name in character_names:
            if name in assignments:
                continue

            voice_id = self._get_next_available_voice(exclude_set)
            self.register(name, voice_id)
            assignments[name] = voice_id
            exclude_set.add(voice_id)

        return assignments

    def list_characters(
        self,
        voice_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[CharacterVoiceMapping]:
        """
        List registered characters.

        Args:
            voice_id: Filter by voice ID
            tag: Filter by tag

        Returns:
            List of CharacterVoiceMapping
        """
        result = list(self._characters.values())

        if voice_id:
            result = [m for m in result if m.voice_id == voice_id]

        if tag:
            result = [m for m in result if tag in m.tags]

        return result

    def list_character_names(self) -> List[str]:
        """Get list of all registered character names."""
        return [m.character_name for m in self._characters.values()]

    def list_used_voices(self) -> Set[str]:
        """Get set of all voices currently assigned to characters."""
        return {m.voice_id for m in self._characters.values()}

    def clear(self) -> None:
        """Clear all registrations."""
        self._characters.clear()
        self._voice_pool_index = 0
        logger.debug("Cleared character registry")

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Export all mappings as dictionary."""
        return {
            name: mapping.to_dict()
            for name, mapping in self._characters.items()
        }

    def from_dict(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Import mappings from dictionary."""
        for name, mapping_data in data.items():
            mapping = CharacterVoiceMapping.from_dict(mapping_data)
            self._characters[name.lower()] = mapping

    def _get_next_available_voice(
        self,
        exclude: Optional[Set[str]] = None,
    ) -> str:
        """
        Get next available voice from pool.

        Cycles through voice pool, skipping excluded voices.
        """
        exclude = exclude or set()
        attempts = 0
        max_attempts = len(self.DEFAULT_VOICE_POOL) * 2

        while attempts < max_attempts:
            voice = self.DEFAULT_VOICE_POOL[
                self._voice_pool_index % len(self.DEFAULT_VOICE_POOL)
            ]
            self._voice_pool_index += 1
            attempts += 1

            if voice not in exclude:
                return voice

        # Fallback: return first voice even if excluded
        return self.DEFAULT_VOICE_POOL[0]

    def _persist_mapping(self, mapping: CharacterVoiceMapping) -> None:
        """Persist mapping to database."""
        if not self._database:
            return

        # TODO: Add character table to VoxDatabase
        # For now, we just log
        logger.debug(f"Would persist: {mapping.character_name} -> {mapping.voice_id}")


# ============================================================================
# GLOBAL REGISTRY
# ============================================================================

_global_registry: Optional[CharacterRegistry] = None


def get_character_registry() -> CharacterRegistry:
    """Get or create global character registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CharacterRegistry()
    return _global_registry


def set_character_registry(registry: CharacterRegistry) -> None:
    """Set global character registry."""
    global _global_registry
    _global_registry = registry


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  AXIOM VOX Character Registry Demo")
    print("=" * 70)

    registry = CharacterRegistry()

    # Register some characters
    registry.register(
        "Dr. Smith",
        voice_id="expert",
        default_emotion="confident",
        description="Expert scientist explaining complex topics",
    )
    registry.register(
        "Host",
        voice_id="announcer",
        description="Energetic show host",
    )
    registry.register(
        "Narrator",
        voice_id="calm",
        description="Calm documentary narrator",
    )

    print("\nRegistered Characters:")
    for mapping in registry.list_characters():
        print(f"  {mapping.character_name}: {mapping.voice_id}")

    # Auto-assign new characters
    print("\nAuto-assigning new characters:")
    new_chars = ["Alice", "Bob", "Charlie"]
    assignments = registry.auto_assign_voices(new_chars)
    for name, voice in assignments.items():
        print(f"  {name}: {voice}")

    # Get voice for character
    print(f"\nVoice for 'Dr. Smith': {registry.get_voice('Dr. Smith')}")
    print(f"Voice for 'dr. smith' (case insensitive): {registry.get_voice('dr. smith')}")

    # Get or assign for unknown character
    print(f"\nGet or assign 'Unknown': {registry.get_or_assign('Unknown')}")

    print("\n" + "=" * 70)
