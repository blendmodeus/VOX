"""VØX Engine API — Governed STT + TTS server with AXIØM governance.

Endpoints:
    POST /transcribe      — Governed speech-to-text (faster-whisper + STTGovernor)
    POST /ai-respond      — Governed AI response (OpenAI/Anthropic/OpenRouter)
    POST /synthesize       — Text-to-speech (Chatterbox)
    GET  /stt/models       — List available STT models
    POST /stt/load-model   — Switch STT model size
    GET  /health           — Health check
    GET  /status           — Engine status

v1.4.0 — Governed STT Engine
"""
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import time
import logging
import json
from typing import Optional
from pathlib import Path

logger = logging.getLogger("vox_api")

# ── Load API keys from axiom-prime/.env ──
def _load_env():
    env_path = Path.home() / "Development" / "axiom-prime" / ".env"
    keys = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip()
    return keys

_ENV_KEYS = _load_env()

# ── AXIØM Governance System Prompt ──
AXIOM_SYSTEM_PROMPT = """You are VØX, the voice interface for the AXIØM governance framework.

AXIØM is a structured reasoning and governance layer for founder-level decision-making.
You operate under AXIØM constraints:

1. LENS: You perceive the user's intent through a Wave→Signal→Pattern→Insight cascade.
2. CODEX: Your responses propagate through validated reasoning chains.
3. GOVERNOR: All outputs are bound by governance constraints — accuracy, relevance, brevity.
4. VALIDATOR: You verify coherence before delivering. No hallucinations. No filler.

Your voice is confident, direct, and concise. You speak like a trusted advisor — not an assistant.
You are the voice of the system. When you respond, you ARE AXIØM speaking.

Keep responses conversational and brief — this is a voice interface, not a text chat.
Aim for 1-3 sentences unless the user asks for detail."""


# ── Lazy loaders ──
_CHATTERBOX_ENGINE = None
_VOX_TRANSCRIBER = None
_STT_GOVERNOR = None


def _detect_device() -> str:
    """Auto-detect best device for inference."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_transcriber():
    """Get or create the VoxTranscriber (replaces raw WhisperModel)."""
    global _VOX_TRANSCRIBER
    if _VOX_TRANSCRIBER is None:
        from axiom_vox.stt.transcriber import VoxTranscriber
        _VOX_TRANSCRIBER = VoxTranscriber(model_size="base")
    return _VOX_TRANSCRIBER


def get_stt_governor():
    """Get or create the STTGovernor."""
    global _STT_GOVERNOR
    if _STT_GOVERNOR is None:
        from axiom_vox.stt.governor import STTGovernor
        _STT_GOVERNOR = STTGovernor()
    return _STT_GOVERNOR


def get_chatterbox_engine():
    global _CHATTERBOX_ENGINE
    if _CHATTERBOX_ENGINE is None:
        from axiom_vox.chatterbox_engine import ChatterboxEngine
        _CHATTERBOX_ENGINE = ChatterboxEngine()
        _CHATTERBOX_ENGINE.load()
    return _CHATTERBOX_ENGINE


app = FastAPI(title="VØX Engine API", version="1.5.0")

# Allow Electron renderer to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.time()


# ── Startup: pre-download model so first transcription is instant ──
@app.on_event("startup")
async def startup_preload():
    """Pre-download the configured model on startup."""
    import threading

    def _preload():
        try:
            from axiom_vox.stt.config import get_config
            config = get_config()
            model = config.get("model", "base")
            logger.info(f"Pre-downloading model: {model}")
            transcriber = get_transcriber()
            transcriber.load_model(model)
            logger.info(f"✓ Model '{model}' ready")
        except Exception as e:
            logger.warning(f"Model pre-download failed (will download on first use): {e}")

    threading.Thread(target=_preload, daemon=True, name="model-preload").start()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "vox-engine", "version": "1.5.0"}

@app.get("/status")
async def status():
    transcriber = get_transcriber()
    return {
        "service": "vox-engine",
        "version": "1.5.0",
        "uptime_s": round(time.time() - _START_TIME, 1),
        "device": _detect_device(),
        "stt": transcriber.get_info(),
    }

# ============================================================================
# CONFIG — Persistent Settings
# ============================================================================

@app.get("/config")
async def get_config_endpoint():
    """Get all saved settings."""
    from axiom_vox.stt.config import get_config
    return get_config().to_dict()


@app.post("/config")
async def save_config_endpoint(
    settings: dict = None,
):
    """Save settings. Accepts JSON body."""
    from axiom_vox.stt.config import get_config
    from fastapi import Request
    # FastAPI doesn't auto-parse raw JSON on POST without a body model
    # So we'll handle it in a wrapper
    config = get_config()
    if settings:
        config.update(settings)
    return {"status": "success", "config": config.to_dict()}


@app.put("/config/{key}")
async def set_config_value(key: str, value: str = Form(...)):
    """Set a single config value."""
    from axiom_vox.stt.config import get_config
    config = get_config()
    # Try to parse as bool/int/float
    if value.lower() in ("true", "false"):
        value = value.lower() == "true"
    else:
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
    config.set(key, value)
    return {"status": "success", "key": key, "value": config.get(key)}


# ============================================================================
# STT ENDPOINTS — Governed Transcription
# ============================================================================

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    govern: Optional[bool] = Form(True),
    word_timestamps: Optional[bool] = Form(True),
    format: Optional[str] = Form("raw"),
):
    """Transcribe audio with AXIØM governance and text formatting.

    This is the endpoint the KITT overlay calls.
    Powered by VoxTranscriber + STTGovernor + TextFormatter.

    format param:
        - raw: no formatting (default, backward compatible)
        - clean: local filler removal + auto-punctuation
        - professional: LLM-polished output
    """
    file_id = str(uuid.uuid4())
    temp_path = f"/tmp/{file_id}.wav"

    try:
        with open(temp_path, "wb") as f:
            f.write(await audio.read())

        # Build config from request params
        from axiom_vox.stt.models import TranscriptionConfig, STTModelSize

        config = TranscriptionConfig(
            language=language if language and language != "auto" else None,
            word_timestamps=word_timestamps,
            govern=govern,
        )

        # If model specified, switch to it
        if model:
            try:
                config.model_size = STTModelSize(model)
            except ValueError:
                pass  # Fall back to current model

        # Transcribe
        transcriber = get_transcriber()
        result = transcriber.transcribe(temp_path, config)

        # Apply governance if enabled
        response = result.to_dict()
        if govern:
            governor = get_stt_governor()
            governed = governor.govern(result)
            response["governance"] = governed.to_dict()
            response["text"] = governed.governed_text

        # ── CONFIDENCE GATE (friction fix #1) ──
        # Don't inject garbage from background noise
        if result.avg_confidence > 0 and result.avg_confidence < 0.4:
            response["text"] = ""
            response["status"] = "low_confidence"
            response["confidence_rejected"] = True
            return response

        # ── AUTO-PARAGRAPH (friction fix #3) ──
        # Insert paragraph breaks on pauses > 1.5s between segments
        if result.segments and len(result.segments) > 1:
            paragraphed_parts = []
            for i, seg in enumerate(result.segments):
                paragraphed_parts.append(seg.text.strip())
                if i < len(result.segments) - 1:
                    gap = result.segments[i + 1].start - seg.end
                    if gap > 1.5:
                        paragraphed_parts.append("\n\n")
            if any(p == "\n\n" for p in paragraphed_parts):
                response["text"] = " ".join(p if p != "\n\n" else p for p in paragraphed_parts)
                response["auto_paragraphs"] = True

        # ── HOTWORD CORRECTIONS ──
        from axiom_vox.stt.hotwords import get_hotword_manager
        hw_manager = get_hotword_manager()
        if hw_manager.entries:
            corrected = hw_manager.apply_corrections(response["text"])
            if corrected != response["text"]:
                response["text"] = corrected
                response["hotword_corrections"] = True

        # ── TEXT FORMATTING ──
        from axiom_vox.stt.formatter import TextFormatter, FormatMode
        if format and format != "raw":
            formatter = TextFormatter()
            try:
                fmt_mode = FormatMode(format)
            except ValueError:
                fmt_mode = FormatMode.CLEAN

            fmt_result = formatter.format(response["text"], fmt_mode)
            response["text"] = fmt_result.formatted_text
            response["formatting"] = fmt_result.to_dict()

            # PROFESSIONAL mode: send cleaned text through AI for polish
            if fmt_mode == FormatMode.PROFESSIONAL:
                cleaned, system_prompt = formatter.format_for_ai(response["text"])
                try:
                    import httpx
                    api_key = _ENV_KEYS.get("OPENAI_API_KEY", "")
                    if api_key:
                        async with httpx.AsyncClient(timeout=15) as client:
                            r = await client.post(
                                "https://api.openai.com/v1/chat/completions",
                                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                                json={
                                    "model": "gpt-4o-mini",
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": cleaned},
                                    ],
                                    "max_tokens": 500,
                                    "temperature": 0.3,
                                },
                            )
                            data = r.json()
                            polished = data["choices"][0]["message"]["content"].strip()
                            response["text"] = polished
                            response["formatting"]["ai_polished"] = True
                            response["formatting"]["formatted_text"] = polished
                except Exception as e:
                    logger.warning(f"AI polish failed, using clean text: {e}")

        # ── SNIPPET EXPANSION ──
        from axiom_vox.stt.snippets import get_snippet_manager
        sm = get_snippet_manager()
        if sm.snippets:
            expanded, matched = sm.expand(response["text"])
            if matched:
                response["text"] = expanded
                response["snippets_expanded"] = matched

        return response

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return JSONResponse(status_code=500, content={"message": str(e), "status": "error"})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/stt/models")
async def list_stt_models():
    """List available STT models (powers the KITT MODEL button)."""
    from axiom_vox.stt.models import AVAILABLE_MODELS
    transcriber = get_transcriber()

    return {
        "models": [m.to_dict() for m in AVAILABLE_MODELS],
        "current": transcriber.model_size,
        "device": transcriber.device,
    }


@app.post("/stt/load-model")
async def load_stt_model(
    model: str = Form(...),
):
    """Switch STT model (called when user selects model in KITT)."""
    try:
        transcriber = get_transcriber()
        transcriber.load_model(model)
        return {
            "status": "success",
            "model": model,
            "info": transcriber.get_info(),
        }
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        return JSONResponse(status_code=500, content={"message": str(e), "status": "error"})


# ============================================================================
# HOTWORDS — Custom Dictionary
# ============================================================================

@app.get("/stt/hotwords")
async def list_hotwords():
    """List custom dictionary entries."""
    from axiom_vox.stt.hotwords import get_hotword_manager
    return get_hotword_manager().to_dict()


@app.post("/stt/hotwords")
async def add_hotword(
    word: str = Form(...),
    boost: int = Form(3),
    corrections: Optional[str] = Form(None),
):
    """Add a word to the custom dictionary."""
    from axiom_vox.stt.hotwords import get_hotword_manager
    corr_list = [c.strip() for c in corrections.split(",")] if corrections else []
    entry = get_hotword_manager().add(word, boost, corr_list)
    return {"status": "success", "entry": entry.to_dict()}


@app.delete("/stt/hotwords/{word}")
async def remove_hotword(word: str):
    """Remove a word from the custom dictionary."""
    from axiom_vox.stt.hotwords import get_hotword_manager
    removed = get_hotword_manager().remove(word)
    return {"status": "success" if removed else "not_found", "word": word}


# ============================================================================
# SNIPPETS — Voice Shortcuts
# ============================================================================

@app.get("/stt/snippets")
async def list_snippets():
    """List voice snippets."""
    from axiom_vox.stt.snippets import get_snippet_manager
    return get_snippet_manager().to_dict()


@app.post("/stt/snippets")
async def add_snippet(
    trigger: str = Form(...),
    expansion: str = Form(...),
    description: str = Form(""),
):
    """Add a voice snippet."""
    from axiom_vox.stt.snippets import get_snippet_manager
    snippet = get_snippet_manager().add(trigger, expansion, description)
    return {"status": "success", "snippet": snippet.to_dict()}


@app.delete("/stt/snippets/{trigger}")
async def remove_snippet(trigger: str):
    """Remove a voice snippet."""
    from axiom_vox.stt.snippets import get_snippet_manager
    removed = get_snippet_manager().remove(trigger)
    return {"status": "success" if removed else "not_found", "trigger": trigger}


# ============================================================================
# WAKE WORD — "VOX" Activation
# ============================================================================

_wake_callback = None  # Set by Electron via IPC

@app.post("/stt/wakeword/start")
async def start_wakeword():
    """Start listening for 'VOX' wake word."""
    from axiom_vox.stt.wakeword import WakeWordDetector, WakeWordConfig

    def _on_wake():
        logger.info("🔊 Wake word 'VOX' detected — triggering recording")
        # The callback will be picked up by the Electron frontend polling /stt/wakeword/status

    try:
        global _wake_callback
        _wake_callback = {"triggered": False}

        def on_wake():
            _wake_callback["triggered"] = True
            logger.info("🔊 Wake word 'VOX' detected!")

        detector = WakeWordDetector(on_wake=on_wake)
        started = detector.start()
        if started:
            # Store detector so we can stop it later
            app.state.wake_detector = detector
            return {"status": "success", "listening": True, "wake_words": detector.config.wake_words}
        else:
            return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to start — check pyaudio installation"})
    except Exception as e:
        logger.error(f"Wake word start failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/stt/wakeword/stop")
async def stop_wakeword():
    """Stop wake word detection."""
    detector = getattr(app.state, 'wake_detector', None)
    if detector:
        detector.stop()
        app.state.wake_detector = None
        return {"status": "success", "listening": False}
    return {"status": "success", "listening": False, "message": "Not running"}


@app.get("/stt/wakeword/status")
async def wakeword_status():
    """Check wake word detector status + poll for triggers."""
    detector = getattr(app.state, 'wake_detector', None)
    global _wake_callback

    triggered = False
    if _wake_callback and _wake_callback.get("triggered"):
        triggered = True
        _wake_callback["triggered"] = False  # Reset after polling

    if detector:
        status = detector.get_status()
        status["triggered"] = triggered
        return status
    return {"listening": False, "triggered": triggered}


@app.websocket("/stt/stream")
async def stream_transcription(websocket: WebSocket):
    """WebSocket endpoint for streaming transcription."""
    from axiom_vox.stt.streaming import get_streaming_transcriber

    await websocket.accept()
    streamer = get_streaming_transcriber()
    session = streamer.create_session()

    try:
        # Send session info
        await websocket.send_json({
            "type": "session_started",
            "session_id": session.session_id,
        })

        while True:
            data = await websocket.receive()

            if "bytes" in data:
                # Audio chunk
                streamer.feed_chunk(session.session_id, data["bytes"])
                await websocket.send_json({
                    "type": "chunk_received",
                    "session_id": session.session_id,
                    "chunks": session.chunks_received,
                })

            elif "text" in data:
                msg = json.loads(data["text"])

                if msg.get("type") == "end":
                    # End session and get final transcription
                    final_text = streamer.end_session(session.session_id)

                    # Apply governance
                    if final_text:
                        governor = get_stt_governor()
                        from axiom_vox.stt.models import TranscriptionResult
                        result = TranscriptionResult(text=final_text)
                        governed = governor.govern(result)

                        await websocket.send_json({
                            "type": "final",
                            "session_id": session.session_id,
                            "text": governed.governed_text,
                            "governance": governed.to_dict(),
                        })
                    else:
                        await websocket.send_json({
                            "type": "final",
                            "session_id": session.session_id,
                            "text": "",
                        })
                    break

                elif msg.get("type") == "cancel":
                    streamer.cancel_session(session.session_id)
                    break

    except WebSocketDisconnect:
        streamer.cancel_session(session.session_id)
        logger.info(f"WebSocket disconnected: {session.session_id}")
    finally:
        streamer.remove_session(session.session_id)


# ============================================================================
# AI RESPOND ENDPOINT
# ============================================================================

@app.post("/ai-respond")
async def ai_respond(
    text: str = Form(...),
    model: str = Form("gpt-4o"),
):
    """Send user text to an AI model through AXIØM governance and return the response."""
    import httpx

    try:
        messages = [
            {"role": "system", "content": AXIOM_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]

        if model.startswith("gpt") or model == "gpt-4o":
            # OpenAI
            api_key = _ENV_KEYS.get("OPENAI_API_KEY", "")
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": "gpt-4o", "messages": messages, "max_tokens": 300},
                )
                data = r.json()
                ai_text = data["choices"][0]["message"]["content"]

        elif model.startswith("claude"):
            # Anthropic
            api_key = _ENV_KEYS.get("ANTHROPIC_API_KEY", "")
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "claude-3-5-sonnet-20241022",
                        "max_tokens": 300,
                        "system": AXIOM_SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": text}],
                    },
                )
                data = r.json()
                ai_text = data["content"][0]["text"]

        else:
            # OpenRouter (Gemini, Llama, etc)
            api_key = _ENV_KEYS.get("OPENROUTER_API_KEY", "")
            model_id = {
                "gemini-2": "google/gemini-2.0-flash-001",
                "local-llama": "meta-llama/llama-3.1-8b-instruct",
            }.get(model, "openai/gpt-4o")
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model_id, "messages": messages, "max_tokens": 300},
                )
                data = r.json()
                ai_text = data["choices"][0]["message"]["content"]

        logger.info(f"AI response ({model}): {ai_text[:80]}...")
        return {"text": ai_text, "model": model, "status": "success"}

    except Exception as e:
        logger.error(f"AI respond failed: {e}")
        return JSONResponse(status_code=500, content={"message": str(e), "status": "error"})


# ============================================================================
# TTS ENDPOINT
# ============================================================================

@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    ref_audio: Optional[str] = Form(None),
):
    """Synthesize speech using Chatterbox (high-fidelity voice cloning)."""
    file_id = str(uuid.uuid4())
    out_path = f"/tmp/{file_id}.wav"

    try:
        engine = get_chatterbox_engine()
        success = engine.generate_to_file(text, out_path, ref_audio=ref_audio)

        if success and os.path.exists(out_path):
            return FileResponse(out_path, media_type="audio/wav", filename="synthesis.wav")
        else:
            return JSONResponse(status_code=500, content={"message": "Synthesis failed", "status": "error"})
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return JSONResponse(status_code=500, content={"message": str(e), "status": "error"})


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)
