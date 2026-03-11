"""
Biometric Voice Router
----------------------

Intelligent voice selection using biometric verification,
context analysis, and voice space matching.

Routes synthesis requests to:
    - Explicitly specified voices
    - Biometrically verified voices
    - Context-matched voices
    - Cloned voices with verification
    - Default fallback voices

AXIØM Phase 6: System - "Integrate the parts"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from .models import (
    VoiceRouteType,
    RouteResult,
    VoiceProfile,
    PipelineConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class RouteCandidate:
    """A candidate voice for routing."""
    voice_id: str
    score: float
    route_type: VoiceRouteType
    verified: bool = False
    adapter_path: Optional[str] = None
    reason: str = ""


class BiometricVoiceRouter:
    """
    Routes voice synthesis to appropriate voice based on:
        - Explicit voice_id
        - Speaker embedding match
        - Content context
        - Voice space matching
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        db=None,
    ):
        """
        Initialize voice router.

        Args:
            config: Pipeline configuration
            db: VoxDatabase instance
        """
        self.config = config or PipelineConfig()
        self._db = db
        self._voice_cache: Dict[str, VoiceProfile] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}

    @property
    def db(self):
        """Get database, lazy loading if needed."""
        if self._db is None:
            from ..persistence import get_database
            self._db = get_database()
        return self._db

    async def route(
        self,
        voice_id: Optional[str] = None,
        speaker_embedding: Optional[bytes] = None,
        text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        require_verification: bool = False,
    ) -> RouteResult:
        """
        Route to optimal voice for synthesis.

        Priority:
            1. Explicit voice_id (if valid and verified)
            2. Biometric match (if embedding provided)
            3. Context-based selection
            4. Default voice

        Args:
            voice_id: Explicit voice ID
            speaker_embedding: Speaker embedding for biometric match
            text: Text to synthesize (for context analysis)
            context: Additional context
            require_verification: Require biometric verification

        Returns:
            RouteResult with selected voice
        """
        start_time = time.time()
        candidates: List[RouteCandidate] = []
        context = context or {}

        # 1. Try explicit voice_id
        if voice_id:
            explicit_result = await self._try_explicit_voice(
                voice_id,
                require_verification,
                speaker_embedding,
            )
            if explicit_result:
                candidates.append(explicit_result)

        # 2. Try biometric matching
        if speaker_embedding and self.config.enable_biometric_identification:
            bio_candidates = await self._find_biometric_matches(
                speaker_embedding,
                limit=3,
            )
            candidates.extend(bio_candidates)

        # 3. Try context-based routing
        if text and self.config.enable_context_routing:
            context_candidate = await self._find_context_match(text, context)
            if context_candidate:
                candidates.append(context_candidate)

        # 4. Add default fallback
        candidates.append(RouteCandidate(
            voice_id=self.config.default_voice_id,
            score=0.0,
            route_type=VoiceRouteType.DEFAULT,
            verified=False,
            reason="Default fallback",
        ))

        # Select best candidate
        best = self._select_best_candidate(candidates, require_verification)

        # Build result
        duration_ms = (time.time() - start_time) * 1000

        # Get voice config
        voice_config = await self._get_voice_config(best.voice_id)

        return RouteResult(
            voice_id=best.voice_id,
            route_type=best.route_type,
            voice_verified=best.verified,
            voice_quality_score=best.score,
            adapter_path=best.adapter_path,
            voice_config=voice_config,
            fallback_used=best.route_type == VoiceRouteType.DEFAULT,
            message=best.reason,
        )

    async def _try_explicit_voice(
        self,
        voice_id: str,
        require_verification: bool,
        speaker_embedding: Optional[bytes],
    ) -> Optional[RouteCandidate]:
        """Try to use explicitly specified voice."""
        # Check if voice exists
        voice = await self._get_voice_profile(voice_id)
        if not voice or not voice.is_active:
            logger.debug(f"Voice {voice_id} not found or inactive")
            return None

        verified = False
        score = 0.8  # Base score for explicit selection

        # If it's a cloned voice, may require verification
        if voice.is_cloned and self.config.require_biometric_for_clones:
            if speaker_embedding and voice.biometric_template_id:
                verified, similarity = await self._verify_biometric(
                    speaker_embedding,
                    voice.biometric_template_id,
                )
                if verified:
                    score = similarity
                elif require_verification:
                    logger.warning(f"Voice {voice_id} failed biometric verification")
                    return None

        # Check if biometric enrolled
        if voice.is_biometric_enrolled:
            if speaker_embedding:
                verified, similarity = await self._verify_biometric(
                    speaker_embedding,
                    voice.biometric_template_id,
                )
                score = similarity if verified else score * 0.5

        route_type = VoiceRouteType.CLONED if voice.is_cloned else VoiceRouteType.EXPLICIT

        return RouteCandidate(
            voice_id=voice_id,
            score=score,
            route_type=route_type,
            verified=verified,
            adapter_path=voice.adapter_path,
            reason=f"Explicit selection{' (verified)' if verified else ''}",
        )

    async def _find_biometric_matches(
        self,
        speaker_embedding: bytes,
        limit: int = 3,
    ) -> List[RouteCandidate]:
        """Find voices matching speaker embedding."""
        candidates = []

        try:
            from ..biometrics import (
                get_biometric_storage,
                SpectralFingerprint,
                deserialize_embedding,
            )

            storage = get_biometric_storage()
            fp = SpectralFingerprint()

            # Get query embedding
            query_emb = deserialize_embedding(speaker_embedding)

            # Get all enrolled voices
            enrolled = storage.list_enrolled_voices(limit=100)

            matches = []
            for entry in enrolled:
                template = storage.get_template(entry["voice_id"])
                if template and template.embedding:
                    template_emb = deserialize_embedding(template.embedding)
                    similarity = fp.similarity(query_emb, template_emb)

                    if similarity >= self.config.biometric_similarity_threshold:
                        matches.append((entry["voice_id"], similarity, template))

            # Sort by similarity
            matches.sort(key=lambda x: x[1], reverse=True)

            for voice_id, similarity, template in matches[:limit]:
                voice = await self._get_voice_profile(voice_id)
                candidates.append(RouteCandidate(
                    voice_id=voice_id,
                    score=similarity,
                    route_type=VoiceRouteType.BIOMETRIC,
                    verified=True,
                    adapter_path=voice.adapter_path if voice else None,
                    reason=f"Biometric match (similarity={similarity:.3f})",
                ))

        except ImportError:
            logger.debug("Biometrics module not available for routing")
        except Exception as e:
            logger.error(f"Biometric matching error: {e}")

        return candidates

    async def _find_context_match(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> Optional[RouteCandidate]:
        """Find voice matching content context."""
        try:
            from ..voice_space import VoiceSpaceDirector

            director = VoiceSpaceDirector()
            result = director.direct(text=text, context=context)

            if result.get("matched_voice_id"):
                voice_id = result["matched_voice_id"]
                score = result.get("match_score", 0.7)

                return RouteCandidate(
                    voice_id=voice_id,
                    score=score,
                    route_type=VoiceRouteType.CONTEXT,
                    verified=False,
                    reason=f"Context match (score={score:.3f})",
                )

        except ImportError:
            logger.debug("Voice space module not available for routing")
        except Exception as e:
            logger.debug(f"Context matching error: {e}")

        return None

    async def _verify_biometric(
        self,
        speaker_embedding: bytes,
        template_id: str,
    ) -> Tuple[bool, float]:
        """Verify speaker embedding against template."""
        try:
            from ..biometrics import (
                get_biometric_storage,
                SpectralFingerprint,
                deserialize_embedding,
            )

            storage = get_biometric_storage()
            template = storage.get_template_by_id(template_id)

            if not template:
                return False, 0.0

            fp = SpectralFingerprint()
            query_emb = deserialize_embedding(speaker_embedding)
            template_emb = deserialize_embedding(template.embedding)

            similarity = fp.similarity(query_emb, template_emb)
            verified = similarity >= self.config.biometric_similarity_threshold

            return verified, similarity

        except Exception as e:
            logger.error(f"Biometric verification error: {e}")
            return False, 0.0

    def _select_best_candidate(
        self,
        candidates: List[RouteCandidate],
        require_verification: bool,
    ) -> RouteCandidate:
        """Select best candidate from list."""
        if not candidates:
            return RouteCandidate(
                voice_id=self.config.fallback_voice_id,
                score=0.0,
                route_type=VoiceRouteType.DEFAULT,
                reason="No candidates, using fallback",
            )

        # Filter by verification if required
        if require_verification:
            verified = [c for c in candidates if c.verified]
            if verified:
                candidates = verified
            else:
                # No verified candidates, use fallback
                return RouteCandidate(
                    voice_id=self.config.fallback_voice_id,
                    score=0.0,
                    route_type=VoiceRouteType.DEFAULT,
                    reason="No verified candidates, using fallback",
                )

        # Sort by score (with type priority)
        type_priority = {
            VoiceRouteType.EXPLICIT: 1.0,
            VoiceRouteType.BIOMETRIC: 0.9,
            VoiceRouteType.CLONED: 0.8,
            VoiceRouteType.CONTEXT: 0.7,
            VoiceRouteType.DEFAULT: 0.0,
        }

        def candidate_key(c: RouteCandidate) -> float:
            return c.score * type_priority.get(c.route_type, 0.5)

        candidates.sort(key=candidate_key, reverse=True)
        return candidates[0]

    async def _get_voice_profile(self, voice_id: str) -> Optional[VoiceProfile]:
        """Get voice profile with caching."""
        if voice_id in self._voice_cache:
            return self._voice_cache[voice_id]

        try:
            # Get from database
            voice_info = self.db.get_voice(voice_id)
            if not voice_info:
                return None

            # Check biometric enrollment
            biometric_template_id = None
            is_biometric_enrolled = False
            try:
                from ..biometrics import get_biometric_storage
                storage = get_biometric_storage()
                template = storage.get_template(voice_id)
                if template:
                    biometric_template_id = template.template_id
                    is_biometric_enrolled = True
            except Exception:
                pass

            # Check for LoRA adapter
            adapter_id = None
            adapter_path = None
            is_cloned = False
            try:
                adapter = self.db.conn.execute(
                    "SELECT * FROM lora_adapters WHERE voice_id = ? AND is_active = 1",
                    (voice_id,)
                ).fetchone()
                if adapter:
                    adapter_id = adapter["adapter_id"]
                    adapter_path = adapter["adapter_path"]
                    is_cloned = True
            except Exception:
                pass

            profile = VoiceProfile(
                voice_id=voice_id,
                name=voice_info.get("name", voice_id),
                is_active=True,
                is_cloned=is_cloned,
                is_biometric_enrolled=is_biometric_enrolled,
                biometric_template_id=biometric_template_id,
                adapter_id=adapter_id,
                adapter_path=adapter_path,
                owner_id=voice_info.get("owner_id"),
            )

            self._voice_cache[voice_id] = profile
            return profile

        except Exception as e:
            logger.error(f"Error getting voice profile: {e}")
            return None

    async def _get_voice_config(self, voice_id: str) -> Dict[str, Any]:
        """Get synthesis configuration for voice."""
        config = {
            "voice_id": voice_id,
        }

        profile = await self._get_voice_profile(voice_id)
        if profile:
            config["adapter_path"] = profile.adapter_path
            config["is_cloned"] = profile.is_cloned

        return config

    def register_voice(
        self,
        voice_id: str,
        profile: VoiceProfile,
    ) -> None:
        """Register a voice profile for routing."""
        self._voice_cache[voice_id] = profile

    def invalidate_cache(self, voice_id: Optional[str] = None) -> None:
        """Invalidate voice cache."""
        if voice_id:
            self._voice_cache.pop(voice_id, None)
            self._embedding_cache.pop(voice_id, None)
        else:
            self._voice_cache.clear()
            self._embedding_cache.clear()


# Singleton instance
_router_instance: Optional[BiometricVoiceRouter] = None


def get_voice_router(config: Optional[PipelineConfig] = None) -> BiometricVoiceRouter:
    """Get or create voice router singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = BiometricVoiceRouter(config=config)
    return _router_instance


def set_voice_router(router: BiometricVoiceRouter) -> None:
    """Set the voice router singleton."""
    global _router_instance
    _router_instance = router
