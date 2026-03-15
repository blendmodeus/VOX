FROM python:3.11-slim

WORKDIR /app

# System deps for faster-whisper (ctranslate2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy engine code
COPY engine/ ./engine/

# Pre-download base model on build (avoids first-run latency)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('Model cached')"

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import httpx; r = httpx.get('http://127.0.0.1:8000/health'); assert r.status_code == 200"

CMD ["python", "-m", "uvicorn", "axiom_vox.vox_api:app", "--host", "0.0.0.0", "--port", "8000"]
