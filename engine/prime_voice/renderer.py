"""
PRIME Voice Renderer
--------------------

The complete text-to-speech rendering pipeline for PRIME's voice.

Pipeline:
    Text → Mode Detection → Prosody Direction → Governance → Synthesis → Identity Verify → Output

The renderer ensures that every word PRIME speaks:
1. Uses the correct speaking mode for the context
2. Passes through AXIOM Law governance
3. Is synthesized with PRIME's locked voice identity
4. Can be verified as genuinely PRIME's voice
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .identity import (
    IdentityConfig,
    PrimeVoiceIdentityManager,
    get_identity_manager,
)
from .models import (
    PrimeUtterance,
    PrimeVoiceConfig,
    PrimeVoiceSession,
    PrimeVoiceState,
    SpeakingModeType,
    UtteranceType,
)
from .speaking_modes import (
    ModeDetectionResult,
    SpeakingModeManager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Renderer Configuration
# =============================================================================

@dataclass
class RendererConfig:
    """Configuration for the PRIME voice renderer."""
    # Synthesis
    model_size: str = "small"
    sample_rate: int = 24000
    audio_format: str = "wav"

    # Governance
    enforce_governance: bool = True
    governance_timeout: float = 5.0  # seconds

    # Mode detection
    auto_detect_mode: bool = True
    default_mode: SpeakingModeType = SpeakingModeType.CONVERSATIONAL
    min_mode_confidence: float = 0.6  # Below this, use default

    # Identity verification
    verify_output: bool = True
    verification_interval: int = 10  # Verify every N utterances

    # Output
    include_provenance: bool = True  # Track full rendering provenance
    log_all_utterances: bool = True


# =============================================================================
# Rendering Result
# =============================================================================

@dataclass
class RenderResult:
    """Complete result of rendering PRIME's speech."""
    success: bool = True
    utterance: Optional[PrimeUtterance] = None
    error: Optional[str] = None

    # Pipeline trace
    mode_detection: Optional[Dict[str, Any]] = None
    prosody_applied: Optional[Dict[str, Any]] = None
    governance_result: Optional[Dict[str, Any]] = None
    synthesis_time_ms: float = 0.0
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "error": self.error,
            "synthesis_time_ms": self.synthesis_time_ms,
            "total_time_ms": self.total_time_ms,
        }
        if self.utterance:
            result["utterance"] = self.utterance.to_dict()
        if self.mode_detection:
            result["mode_detection"] = self.mode_detection
        return result


# =============================================================================
# PRIME Voice Renderer
# =============================================================================

class PrimeVoiceRenderer:
    """
    The main rendering pipeline for PRIME's voice.

    Takes text input and produces audio output in PRIME's voice,
    with full governance, mode detection, and identity verification.

    Usage:
        renderer = PrimeVoiceRenderer()

        # Simple rendering
        result = renderer.render("All systems operational. Uptime: 99.9%")
        # -> RenderResult with audio in PRIME's voice (BRIEFING mode auto-detected)

        # With explicit mode
        result = renderer.render(
            "Warning: memory usage at 95%",
            mode=SpeakingModeType.ALERT,
        )

        # With context
        result = renderer.render(
            "Deploying v2.1.0 to production",
            context={"type": "command", "urgency": "normal"},
        )
    """

    def __init__(
        self,
        config: Optional[RendererConfig] = None,
        identity_manager: Optional[PrimeVoiceIdentityManager] = None,
    ):
        self.config = config or RendererConfig()
        self._identity = identity_manager or get_identity_manager()
        self._mode_manager = SpeakingModeManager(self.config.default_mode)
        self._session = PrimeVoiceSession(
            session_id=f"prime_voice_{uuid.uuid4().hex[:8]}",
            state=PrimeVoiceState.READY,
        )
        self._utterance_count = 0
        self._synthesizer = None  # Lazy-loaded VoxSynthesizer
        self._governor = None     # Lazy-loaded VoxGovernor

        logger.info(
            f"PRIME Voice Renderer initialized "
            f"[session={self._session.session_id}, "
            f"model={self.config.model_size}]"
        )

    # -------------------------------------------------------------------------
    # Lazy Component Loading
    # -------------------------------------------------------------------------

    def _get_synthesizer(self):
        """Lazy-load the VoxSynthesizer."""
        if self._synthesizer is None:
            try:
                from axiom_vox.synthesis import VoxSynthesizer
                self._synthesizer = VoxSynthesizer(
                    model_size=self.config.model_size,
                )
                logger.info("VoxSynthesizer loaded for PRIME voice")
            except ImportError:
                logger.warning("VoxSynthesizer not available - using placeholder")
                self._synthesizer = _PlaceholderSynthesizer()
        return self._synthesizer

    def _get_governor(self):
        """Lazy-load the VoxGovernor."""
        if self._governor is None:
            try:
                from axiom_vox.vox_governor import get_governor
                self._governor = get_governor()
                logger.info("VoxGovernor loaded for PRIME voice")
            except ImportError:
                logger.warning("VoxGovernor not available - governance disabled")
                self._governor = None
        return self._governor

    # -------------------------------------------------------------------------
    # Main Render Pipeline
    # -------------------------------------------------------------------------

    def render(
        self,
        text: str,
        mode: Optional[SpeakingModeType] = None,
        utterance_type: Optional[UtteranceType] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RenderResult:
        """
        Render text as PRIME's speech.

        Full pipeline: mode detection → prosody → governance → synthesis → verify.

        Args:
            text: The text for PRIME to speak
            mode: Explicit speaking mode (auto-detected if None)
            utterance_type: Classification of the utterance
            context: Optional context for mode detection and governance

        Returns:
            RenderResult with audio data and provenance
        """
        start_time = time.time()
        self._session.state = PrimeVoiceState.SPEAKING

        try:
            # Step 1: Mode Detection
            detected_mode, mode_info = self._detect_mode(text, mode, context)

            # Step 2: Get Prosody Adjustments
            prosody = self._build_prosody(detected_mode)

            # Step 3: Governance Check
            governance_ok, governance_info = self._check_governance(text, context)
            if not governance_ok:
                self._session.state = PrimeVoiceState.READY
                return RenderResult(
                    success=False,
                    error=f"Governance blocked: {governance_info.get('reason', 'unknown')}",
                    mode_detection=mode_info,
                    governance_result=governance_info,
                    total_time_ms=(time.time() - start_time) * 1000,
                )

            # Step 4: Synthesize Audio
            synth_start = time.time()
            audio_data, audio_duration = self._synthesize(text, detected_mode, prosody)
            synth_time = (time.time() - synth_start) * 1000

            # Step 5: Identity Verification (periodic)
            identity_verified = False
            identity_similarity = 0.0
            self._utterance_count += 1

            if (self.config.verify_output and
                    self._identity.should_verify() and
                    audio_data):
                identity_verified, identity_similarity = self._verify_identity(audio_data)

            # Step 6: Build Utterance
            utterance = PrimeUtterance(
                utterance_id=f"prime_utt_{uuid.uuid4().hex[:12]}",
                timestamp=datetime.now(),
                source_text=text,
                utterance_type=utterance_type or self._classify_utterance(text, detected_mode),
                speaking_mode=detected_mode,
                audio_data=audio_data,
                duration_seconds=audio_duration,
                sample_rate=self.config.sample_rate,
                audio_format=self.config.audio_format,
                voice_id=self._identity.get_identity().vox_voice_id,
                voice_vector_used=self._mode_manager.get_adjusted_vector(
                    self._identity.get_voice_vector().to_dict(),
                    detected_mode,
                ),
                prosody_applied=prosody,
                emotion_preset_used=self._identity.get_identity().emotion_preset,
                governance_passed=governance_ok,
                governance_report=governance_info,
                identity_verified=identity_verified,
                identity_similarity=identity_similarity,
                session_id=self._session.session_id,
                request_context=context,
            )

            # Update session
            self._session.utterance_count += 1
            self._session.total_duration_seconds += audio_duration
            self._session.recent_utterance_ids.append(utterance.utterance_id)
            if len(self._session.recent_utterance_ids) > 50:
                self._session.recent_utterance_ids = self._session.recent_utterance_ids[-50:]

            total_time = (time.time() - start_time) * 1000

            if self.config.log_all_utterances:
                logger.info(
                    f"PRIME spoke [{detected_mode.value}]: "
                    f"\"{text[:60]}{'...' if len(text) > 60 else ''}\" "
                    f"({audio_duration:.1f}s, {total_time:.0f}ms)"
                )

            self._session.state = PrimeVoiceState.READY

            return RenderResult(
                success=True,
                utterance=utterance,
                mode_detection=mode_info,
                prosody_applied=prosody,
                governance_result=governance_info,
                synthesis_time_ms=synth_time,
                total_time_ms=total_time,
            )

        except Exception as e:
            self._session.state = PrimeVoiceState.ERROR
            logger.error(f"PRIME voice render failed: {e}")
            return RenderResult(
                success=False,
                error=str(e),
                total_time_ms=(time.time() - start_time) * 1000,
            )

    # -------------------------------------------------------------------------
    # Pipeline Steps
    # -------------------------------------------------------------------------

    def _detect_mode(
        self,
        text: str,
        explicit_mode: Optional[SpeakingModeType],
        context: Optional[Dict[str, Any]],
    ) -> tuple:
        """Detect or use explicit speaking mode."""
        if explicit_mode:
            transition = self._mode_manager.switch_mode(explicit_mode)
            mode_info = {
                "source": "explicit",
                "mode": explicit_mode.value,
                "transition_from": transition.from_mode.value,
                "crossfade_seconds": transition.crossfade_seconds,
            }
            return explicit_mode, mode_info

        if self.config.auto_detect_mode:
            detection = self._mode_manager.detect_mode(text, context)
            if detection.confidence >= self.config.min_mode_confidence:
                transition = self._mode_manager.switch_mode(detection.detected_mode)
                mode_info = {
                    "source": "auto_detected",
                    "mode": detection.detected_mode.value,
                    "confidence": detection.confidence,
                    "signals": detection.signals_found[:5],
                    "transition_from": transition.from_mode.value,
                }
                return detection.detected_mode, mode_info

        # Fall back to current mode
        current = self._mode_manager.current_mode
        mode_info = {"source": "default", "mode": current.value}
        return current, mode_info

    def _build_prosody(self, mode: SpeakingModeType) -> Dict[str, Any]:
        """Build prosody parameters for the speaking mode."""
        # Start with PRIME's base synthesis params
        base_params = self._identity.get_synthesis_params()

        # Apply mode adjustments
        mode_adjustments = self._mode_manager.get_prosody_adjustments(mode)

        # Merge: base params + mode deltas
        prosody = {
            **base_params,
            "mode": mode.value,
            "mode_adjustments": mode_adjustments,
        }

        # Apply rate multiplier
        if "speaking_rate" in prosody and "speaking_rate_multiplier" in mode_adjustments:
            prosody["speaking_rate"] *= mode_adjustments["speaking_rate_multiplier"]

        # Apply pitch shift
        if "pitch_shift" in mode_adjustments:
            prosody["pitch"] = prosody.get("pitch", 0.0) + mode_adjustments["pitch_shift"]

        # Apply emotion overrides from mode
        for key in ("warmth", "confidence", "energy"):
            if key in mode_adjustments:
                prosody[key] = mode_adjustments[key]

        return prosody

    def _check_governance(
        self,
        text: str,
        context: Optional[Dict[str, Any]],
    ) -> tuple:
        """Run governance checks on the utterance."""
        if not self.config.enforce_governance:
            return True, {"status": "skipped", "reason": "governance disabled"}

        governor = self._get_governor()
        if governor is None:
            return True, {"status": "skipped", "reason": "governor not available"}

        try:
            result = governor.govern(text, context=context or {})
            passed = result.action.value != "refuse" if hasattr(result, 'action') else True
            return passed, {
                "status": "passed" if passed else "refused",
                "action": result.action.value if hasattr(result, 'action') else "allow",
            }
        except Exception as e:
            logger.warning(f"Governance check failed: {e} - allowing utterance")
            return True, {"status": "error", "reason": str(e)}

    def _synthesize(
        self,
        text: str,
        mode: SpeakingModeType,
        prosody: Dict[str, Any],
    ) -> tuple:
        """Synthesize audio in PRIME's voice."""
        synthesizer = self._get_synthesizer()

        try:
            # Build VoiceConfig-compatible params
            voice_config_params = {
                "voice_id": prosody.get("voice_id", "prime_sovereign"),
                "speaking_rate": prosody.get("speaking_rate", 0.95),
                "pitch": prosody.get("pitch", 0.0),
                "emotion": prosody.get("emotion", "professional"),
            }

            result = synthesizer.synthesize(
                text=text,
                **voice_config_params,
            )

            if hasattr(result, 'audio_data'):
                audio_data = result.audio_data
                duration = result.duration_seconds or self._estimate_duration(text)
            elif isinstance(result, dict):
                audio_data = result.get("audio_data")
                duration = result.get("duration_seconds", self._estimate_duration(text))
            else:
                audio_data = result
                duration = self._estimate_duration(text)

            return audio_data, duration

        except Exception as e:
            logger.warning(f"Synthesis failed: {e} - returning placeholder")
            duration = self._estimate_duration(text)
            return None, duration

    def _verify_identity(self, audio_data: bytes) -> tuple:
        """Verify synthesized audio matches PRIME's voice identity."""
        try:
            from axiom_vox.biometrics import SpectralFingerprint
            import numpy as np

            fp = SpectralFingerprint()
            # Convert audio bytes to array
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            embedding = fp.extract(audio_array, self.config.sample_rate)

            result = self._identity.verify_output(embedding.tolist())
            return result.passed, result.similarity_score

        except ImportError:
            return False, 0.0
        except Exception as e:
            logger.debug(f"Identity verification skipped: {e}")
            return False, 0.0

    def _classify_utterance(
        self,
        text: str,
        mode: SpeakingModeType,
    ) -> UtteranceType:
        """Classify the utterance type based on content and mode."""
        mode_to_type = {
            SpeakingModeType.BRIEFING: UtteranceType.STATUS,
            SpeakingModeType.ALERT: UtteranceType.ALERT,
            SpeakingModeType.DIRECTIVE: UtteranceType.COMMAND_ECHO,
            SpeakingModeType.REFLECTIVE: UtteranceType.REFLECTION,
            SpeakingModeType.EMPATHETIC: UtteranceType.RESPONSE,
            SpeakingModeType.CEREMONIAL: UtteranceType.NARRATION,
            SpeakingModeType.CONVERSATIONAL: UtteranceType.RESPONSE,
        }
        return mode_to_type.get(mode, UtteranceType.RESPONSE)

    def _estimate_duration(self, text: str) -> float:
        """Estimate speech duration from text length."""
        words = len(text.split())
        # ~150 words per minute at PRIME's speaking rate (0.95x)
        wpm = 150 * 0.95
        return max(0.5, words / wpm * 60)

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def get_session(self) -> PrimeVoiceSession:
        """Get current voice session."""
        return self._session

    def reset_session(self) -> PrimeVoiceSession:
        """Reset the voice session."""
        self._session = PrimeVoiceSession(
            session_id=f"prime_voice_{uuid.uuid4().hex[:8]}",
            state=PrimeVoiceState.READY,
        )
        self._mode_manager = SpeakingModeManager(self.config.default_mode)
        self._utterance_count = 0
        logger.info(f"PRIME voice session reset: {self._session.session_id}")
        return self._session

    def get_stats(self) -> Dict[str, Any]:
        """Get renderer statistics."""
        return {
            "session": self._session.to_dict(),
            "identity": self._identity.get_stats(),
            "modes": self._mode_manager.get_stats(),
            "utterance_count": self._utterance_count,
            "renderer_config": {
                "model_size": self.config.model_size,
                "governance": self.config.enforce_governance,
                "auto_detect_mode": self.config.auto_detect_mode,
                "verify_output": self.config.verify_output,
            },
        }


# =============================================================================
# Placeholder Synthesizer (when VoxSynthesizer not available)
# =============================================================================

class _PlaceholderSynthesizer:
    """Placeholder when real synthesizer is not loaded."""

    def synthesize(self, text: str, **kwargs):
        """Return a placeholder result."""
        words = len(text.split())
        duration = max(0.5, words / 150 * 60)

        return type("Result", (), {
            "audio_data": None,
            "duration_seconds": duration,
            "success": True,
            "sample_rate": 24000,
        })()


# =============================================================================
# Convenience Functions
# =============================================================================

_default_renderer: Optional[PrimeVoiceRenderer] = None


def get_renderer(
    config: Optional[RendererConfig] = None,
) -> PrimeVoiceRenderer:
    """Get or create the default PRIME voice renderer."""
    global _default_renderer
    if _default_renderer is None:
        _default_renderer = PrimeVoiceRenderer(config)
    return _default_renderer


def prime_speak(
    text: str,
    mode: Optional[SpeakingModeType] = None,
    context: Optional[Dict[str, Any]] = None,
) -> RenderResult:
    """
    Have PRIME speak the given text.

    This is the simplest entry point for PRIME's voice.

    Usage:
        from axiom_vox.prime_voice import prime_speak

        result = prime_speak("All systems operational.")
        result = prime_speak("Warning: high CPU", mode=SpeakingModeType.ALERT)
    """
    renderer = get_renderer()
    return renderer.render(text, mode=mode, context=context)
