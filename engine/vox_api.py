"""VØX Engine API — Local STT + TTS server with AXIØM AI governance.
Serves /transcribe (faster-whisper), /ai-respond (governed AI), and /synthesize (Chatterbox).
"""
from fastapi import FastAPI, UploadFile, File, Form
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


# Lazy loaders
_WHISPER_MODEL = None
_CHATTERBOX_ENGINE = None

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

def get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel
        device = _detect_device()
        # MPS not yet supported by faster-whisper/CTranslate2 — use CPU with int8
        # When CTranslate2 adds MPS, this will auto-upgrade
        if device == "mps":
            logger.info("MPS detected but CTranslate2 uses CPU. Using int8 quantization.")
            _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
        elif device == "cuda":
            _WHISPER_MODEL = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
        else:
            _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info(f"Whisper model loaded (device={device})")
    return _WHISPER_MODEL

def get_chatterbox_engine():
    global _CHATTERBOX_ENGINE
    if _CHATTERBOX_ENGINE is None:
        from axiom_vox.chatterbox_engine import ChatterboxEngine
        _CHATTERBOX_ENGINE = ChatterboxEngine()
        _CHATTERBOX_ENGINE.load()
    return _CHATTERBOX_ENGINE


app = FastAPI(title="VØX Engine API")

# Allow Electron renderer to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.time()

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "vox-engine"}

@app.get("/status")
async def status():
    return {
        "service": "vox-engine",
        "uptime_s": round(time.time() - _START_TIME, 1),
        "device": _detect_device(),
        "whisper_model": "base",
    }

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    """Transcribe audio using faster-whisper."""
    file_id = str(uuid.uuid4())
    temp_path = f"/tmp/{file_id}.wav"

    try:
        with open(temp_path, "wb") as f:
            f.write(await audio.read())

        t0 = time.time()
        model = get_whisper_model()
        segments, info = model.transcribe(temp_path, language=language)

        full_text = "".join([s.text for s in segments]).strip()
        elapsed = time.time() - t0

        return {
            "text": full_text,
            "language": info.language,
            "duration_s": round(elapsed, 2),
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return JSONResponse(status_code=500, content={"message": str(e), "status": "error"})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
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
