"""
VØX API Server
--------------

FastAPI server that exposes AXIØM VØX text-to-speech capabilities
for the CMNDCNTRL orb and other frontend clients.

Run with:
    python -m axiom_vox.api.server

Or:
    uvicorn axiom_vox.api.server:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import base64
import logging
import asyncio
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import VØX modules
try:
    from axiom_vox.synthesis import VoxSynthesizer, VoiceConfig, AudioFormat, get_synthesizer
    from axiom_vox.vox_governor import VoxGovernor, GovernanceAction
    from axiom_vox.emotion_presets import get_emotion_preset, list_emotion_presets
    HAS_VOX = True
except ImportError as e:
    logger.warning(f"VØX modules not fully available: {e}")
    HAS_VOX = False

# ============================================================================
# Request/Response Models
# ============================================================================

class SynthesizeRequest(BaseModel):
    """Request to synthesize speech."""
    text: str = Field(..., min_length=1, max_length=10000)
    voice_id: str = Field(default="axiom_default")
    emotion_preset: Optional[str] = Field(default="neutral")
    speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=0.0, ge=-1.0, le=1.0)
    volume: float = Field(default=1.0, ge=0.0, le=1.0)
    output_format: str = Field(default="wav")
    stream: bool = Field(default=False)

class SynthesizeResponse(BaseModel):
    """Response from synthesis."""
    success: bool
    audio_base64: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: int = 24000
    format: str = "wav"
    error: Optional[str] = None
    governance_report: Optional[Dict[str, Any]] = None

class VoiceInfo(BaseModel):
    """Information about a voice."""
    voice_id: str
    name: str
    category: str
    description: Optional[str] = None
    supported_emotions: List[str] = []

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    vox_available: bool
    model_loaded: bool

# ============================================================================
# Global State
# ============================================================================

synthesizer: Optional[VoxSynthesizer] = None
governor: Optional[VoxGovernor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup on startup/shutdown."""
    global synthesizer, governor

    logger.info("Starting VØX API Server...")

    if HAS_VOX:
        try:
            synthesizer = get_synthesizer(model_size="small")
            governor = VoxGovernor()
            logger.info("VØX modules initialized")
        except Exception as e:
            logger.error(f"Failed to initialize VØX: {e}")
    else:
        logger.warning("Running in fallback mode (VØX not available)")

    yield

    logger.info("Shutting down VØX API Server...")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="AXIØM VØX API",
    description="Text-to-Speech API for CMNDCNTRL and the Orb",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        vox_available=HAS_VOX,
        model_loaded=synthesizer is not None and synthesizer._loaded if synthesizer else False,
    )

@app.get("/api/vox/voices", response_model=List[VoiceInfo])
async def list_voices():
    """List available voices."""
    # Base voices
    voices = [
        VoiceInfo(
            voice_id="axiom_default",
            name="AXIØM Default",
            category="synthetic",
            description="Default AXIØM voice - clear and professional",
            supported_emotions=["neutral", "calm", "joy", "urgency", "professional", "warm"],
        ),
        VoiceInfo(
            voice_id="axiom_warm",
            name="AXIØM Warm",
            category="synthetic",
            description="Warm and friendly voice for casual interaction",
            supported_emotions=["neutral", "calm", "joy", "warm"],
        ),
        VoiceInfo(
            voice_id="axiom_professional",
            name="AXIØM Professional",
            category="synthetic",
            description="Formal voice for business contexts",
            supported_emotions=["neutral", "professional", "urgency"],
        ),
        VoiceInfo(
            voice_id="axiom_prime",
            name="AXIØM Prime",
            category="synthetic",
            description="The voice of AXIØM Prime - thoughtful and grounded",
            supported_emotions=["neutral", "calm", "joy", "urgency", "professional", "warm", "reflective"],
        ),
    ]

    return voices

@app.get("/api/vox/emotions")
async def list_emotions():
    """List available emotion presets."""
    if HAS_VOX:
        try:
            return list_emotion_presets()
        except Exception:
            pass

    # Fallback list
    return [
        {"id": "neutral", "name": "Neutral", "description": "Default emotional tone"},
        {"id": "calm", "name": "Calm", "description": "Peaceful and relaxed"},
        {"id": "joy", "name": "Joy", "description": "Happy and upbeat"},
        {"id": "urgency", "name": "Urgency", "description": "Alert and pressing"},
        {"id": "professional", "name": "Professional", "description": "Business-like and formal"},
        {"id": "warm", "name": "Warm", "description": "Friendly and welcoming"},
        {"id": "reflective", "name": "Reflective", "description": "Thoughtful and contemplative"},
    ]

@app.post("/api/vox/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text.

    Uses AXIØM VØX with Qwen3-TTS for high-quality synthesis,
    with automatic governance checks.
    """
    # Apply governance
    governed_text = request.text
    governance_report = None

    if governor:
        try:
            result = governor.govern(
                text=request.text,
                voice_id=request.voice_id,
            )

            if result.action == GovernanceAction.REFUSE:
                return SynthesizeResponse(
                    success=False,
                    error=f"Content blocked: {result.refusal_reason}",
                    governance_report=result.to_dict(),
                )

            governed_text = result.governed_text
            governance_report = result.to_dict()
        except Exception as e:
            logger.warning(f"Governance check failed: {e}")

    # Perform synthesis
    if synthesizer:
        try:
            voice_config = VoiceConfig(
                voice_id=request.voice_id,
                speaking_rate=request.speaking_rate,
                pitch=request.pitch,
                volume=request.volume,
                emotion=request.emotion_preset,
            )

            # Map output format
            format_map = {
                "wav": AudioFormat.WAV,
                "mp3": AudioFormat.MP3,
                "ogg": AudioFormat.OGG,
            }
            output_format = format_map.get(request.output_format, AudioFormat.WAV)

            result = synthesizer.synthesize(
                text=governed_text,
                voice=voice_config,
                output_format=output_format,
            )

            if result.success:
                audio_b64 = base64.b64encode(result.audio_data).decode() if result.audio_data else None
                return SynthesizeResponse(
                    success=True,
                    audio_base64=audio_b64,
                    duration_seconds=result.duration_seconds,
                    sample_rate=result.sample_rate,
                    format=request.output_format,
                    governance_report=governance_report,
                )
            else:
                return SynthesizeResponse(
                    success=False,
                    error=result.error,
                    governance_report=governance_report,
                )

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return SynthesizeResponse(
                success=False,
                error=str(e),
                governance_report=governance_report,
            )

    # Fallback: generate placeholder
    return await _synthesize_fallback(request, governed_text, governance_report)

async def _synthesize_fallback(
    request: SynthesizeRequest,
    text: str,
    governance_report: Optional[Dict],
) -> SynthesizeResponse:
    """Fallback synthesis when VØX is not available."""
    try:
        import numpy as np
        import io

        # Try to import soundfile
        try:
            import soundfile as sf
            HAS_SF = True
        except ImportError:
            HAS_SF = False

        if not HAS_SF:
            return SynthesizeResponse(
                success=False,
                error="Audio libraries not available. Install: pip install soundfile numpy",
                governance_report=governance_report,
            )

        # Generate placeholder audio (beep + silence)
        sample_rate = 24000
        duration = min(len(text) / 15, 10.0)  # Rough estimate

        t = np.linspace(0, duration, int(sample_rate * duration))

        # Start beep
        beep_duration = 0.1
        beep_samples = int(sample_rate * beep_duration)
        audio = np.zeros_like(t, dtype=np.float32)
        audio[:beep_samples] = 0.3 * np.sin(2 * np.pi * 440 * t[:beep_samples])

        # Fade out
        fade_samples = int(sample_rate * 0.02)
        if beep_samples > fade_samples:
            audio[beep_samples-fade_samples:beep_samples] *= np.linspace(1, 0, fade_samples)

        # Write to buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        audio_data = buffer.getvalue()

        return SynthesizeResponse(
            success=True,
            audio_base64=base64.b64encode(audio_data).decode(),
            duration_seconds=duration,
            sample_rate=sample_rate,
            format="wav",
            error="PLACEHOLDER: Real TTS model not loaded",
            governance_report=governance_report,
        )

    except Exception as e:
        logger.error(f"Fallback synthesis error: {e}")
        return SynthesizeResponse(
            success=False,
            error=f"Fallback synthesis failed: {e}",
            governance_report=governance_report,
        )

@app.post("/api/vox/synthesize/stream")
async def synthesize_stream(request: SynthesizeRequest):
    """
    Stream synthesized audio chunks.

    Returns audio as chunked transfer encoding for real-time playback.
    """
    if not synthesizer:
        raise HTTPException(
            status_code=503,
            detail="Streaming not available - synthesizer not loaded",
        )

    async def generate():
        """Generate audio chunks."""
        voice_config = VoiceConfig(
            voice_id=request.voice_id,
            speaking_rate=request.speaking_rate,
            pitch=request.pitch,
            volume=request.volume,
            emotion=request.emotion_preset,
        )

        try:
            async for chunk in synthesizer.synthesize_stream(
                text=request.text,
                voice=voice_config,
                chunk_size=4096,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")

    return StreamingResponse(
        generate(),
        media_type="audio/wav",
        headers={
            "Transfer-Encoding": "chunked",
            "X-VØX-Voice": request.voice_id,
        },
    )

@app.post("/api/vox/check")
async def check_content(request: SynthesizeRequest):
    """
    Pre-flight check content against governance rules.

    Returns governance result without synthesizing audio.
    """
    if not governor:
        return {
            "approved": True,
            "action": "allow",
            "message": "Governance not configured",
        }

    try:
        result = governor.govern(
            text=request.text,
            voice_id=request.voice_id,
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Governance check error: {e}")
        return {
            "approved": True,
            "action": "allow",
            "error": str(e),
        }

# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Run the server."""
    import uvicorn

    port = int(os.environ.get("VOX_PORT", "8080"))
    host = os.environ.get("VOX_HOST", "0.0.0.0")

    print("=" * 60)
    print("  AXIØM VØX API Server")
    print("=" * 60)
    print(f"\n  Starting on http://{host}:{port}")
    print(f"  VØX Available: {HAS_VOX}")
    print("\n  Endpoints:")
    print(f"    GET  /health")
    print(f"    GET  /api/vox/voices")
    print(f"    GET  /api/vox/emotions")
    print(f"    POST /api/vox/synthesize")
    print(f"    POST /api/vox/synthesize/stream")
    print(f"    POST /api/vox/check")
    print("\n" + "=" * 60 + "\n")

    uvicorn.run(
        "axiom_vox.api.server:app",
        host=host,
        port=port,
        reload=os.environ.get("VOX_RELOAD", "").lower() == "true",
    )

if __name__ == "__main__":
    main()
