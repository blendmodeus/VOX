"""
VØX Wake Word Detection
-----------------------

Always-on "VOX" wake word — no hotkey needed.

Architecture:
    1. Background thread with mic always listening
    2. Silero VAD detects voice activity (nearly zero CPU when silent)
    3. When voice detected, buffer 1.5s of audio
    4. Run faster-whisper tiny on the buffer (~50ms on CPU)
    5. If transcription contains "vox" → fire callback
    6. If not → discard buffer, keep listening

Resource usage:
    - Silent room: <1% CPU (VAD only)
    - Someone speaking: brief spike for Whisper tiny (~50ms per check)
    - Memory: ~75MB for Whisper tiny model

Usage:
    from axiom_vox.stt.wakeword import WakeWordDetector

    def on_wake():
        print("VOX activated!")

    detector = WakeWordDetector(on_wake=on_wake)
    detector.start()  # Starts background listening
    detector.stop()   # Stops listening
"""

from __future__ import annotations

import io
import os
import wave
import struct
import threading
import logging
import time
from typing import Callable, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""
    wake_words: list = None           # Words to listen for (default: ["vox"])
    model_size: str = "tiny"          # Whisper model for keyword detection
    sample_rate: int = 16000          # Audio sample rate
    buffer_seconds: float = 1.5       # Audio buffer length for detection
    cooldown_seconds: float = 2.0     # Min time between activations
    vad_threshold: float = 0.3        # Voice activity threshold (0-1)
    energy_threshold: int = 300       # RMS energy threshold for speech
    check_interval: float = 0.5      # How often to check buffer (seconds)

    def __post_init__(self):
        if self.wake_words is None:
            self.wake_words = ["vox"]


class WakeWordDetector:
    """Always-on wake word detection using VAD + Whisper tiny.

    Listens continuously in a background thread. When voice activity
    is detected, buffers audio and runs Whisper tiny to check for
    the wake word. Fires callback on detection.
    """

    def __init__(
        self,
        on_wake: Callable[[], None],
        config: Optional[WakeWordConfig] = None,
    ):
        self.on_wake = on_wake
        self.config = config or WakeWordConfig()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._transcriber = None
        self._last_trigger_time = 0.0
        self._audio_interface = None
        self._stream = None

    @property
    def is_listening(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Start listening for wake word in background thread."""
        if self._running:
            logger.warning("Wake word detector already running")
            return False

        if not HAS_NUMPY:
            logger.error("numpy required for wake word detection (pip install numpy)")
            return False

        if not HAS_PYAUDIO:
            logger.error("pyaudio required for wake word detection (pip install pyaudio)")
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="vox-wakeword",
        )
        self._thread.start()
        logger.info(f"🎤 Wake word detector started — listening for: {self.config.wake_words}")
        return True

    def stop(self):
        """Stop listening."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._audio_interface:
            try:
                self._audio_interface.terminate()
            except Exception:
                pass
            self._audio_interface = None
        logger.info("Wake word detector stopped")

    def _get_transcriber(self):
        """Lazy-load a tiny Whisper model for wake word detection."""
        if self._transcriber is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading wake word model: {self.config.model_size}")
                self._transcriber = WhisperModel(
                    self.config.model_size,
                    device="cpu",
                    compute_type="int8",
                )
                logger.info("Wake word model loaded")
            except Exception as e:
                logger.error(f"Failed to load wake word model: {e}")
                raise
        return self._transcriber

    def _listen_loop(self):
        """Main listening loop — runs in background thread."""
        try:
            self._audio_interface = pyaudio.PyAudio()
            chunk_size = int(self.config.sample_rate * self.config.check_interval)
            buffer_chunks = int(self.config.buffer_seconds / self.config.check_interval)

            self._stream = self._audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
            )

            audio_buffer: List[bytes] = []
            voice_active = False

            logger.info("Wake word detector: listening...")

            while self._running:
                try:
                    data = self._stream.read(chunk_size, exception_on_overflow=False)
                except Exception as e:
                    logger.warning(f"Audio read error: {e}")
                    time.sleep(0.1)
                    continue

                # Simple energy-based VAD
                samples = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

                if rms > self.config.energy_threshold:
                    # Voice detected — buffer audio
                    voice_active = True
                    audio_buffer.append(data)

                    # Keep buffer at target length
                    if len(audio_buffer) > buffer_chunks:
                        audio_buffer = audio_buffer[-buffer_chunks:]

                elif voice_active and len(audio_buffer) > 0:
                    # Voice stopped — check buffer for wake word
                    voice_active = False
                    self._check_buffer(audio_buffer)
                    audio_buffer = []

        except Exception as e:
            logger.error(f"Wake word listener error: {e}")
        finally:
            self._running = False

    def _check_buffer(self, audio_chunks: List[bytes]):
        """Check audio buffer for wake word using Whisper tiny."""
        # Cooldown check
        now = time.time()
        if now - self._last_trigger_time < self.config.cooldown_seconds:
            return

        try:
            # Convert audio buffer to WAV
            audio_data = b"".join(audio_chunks)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio_data)

            wav_buffer.seek(0)

            # Transcribe with Whisper tiny
            model = self._get_transcriber()
            segments, info = model.transcribe(
                wav_buffer,
                language="en",
                beam_size=1,        # Fast, not accurate — we just need keyword
                best_of=1,
                vad_filter=False,   # We already did VAD
            )

            text = " ".join(seg.text for seg in segments).lower().strip()

            if not text:
                return

            logger.debug(f"Wake word check: '{text}'")

            # Check for wake words
            for wake_word in self.config.wake_words:
                if wake_word.lower() in text:
                    logger.info(f"🔊 Wake word detected: '{wake_word}' in '{text}'")
                    self._last_trigger_time = now
                    # Fire callback in separate thread to not block listener
                    threading.Thread(
                        target=self.on_wake,
                        daemon=True,
                        name="vox-wake-callback",
                    ).start()
                    return

        except Exception as e:
            logger.warning(f"Wake word check failed: {e}")

    def get_status(self) -> dict:
        """Get detector status."""
        return {
            "listening": self._running,
            "wake_words": self.config.wake_words,
            "model": self.config.model_size,
            "cooldown": self.config.cooldown_seconds,
            "energy_threshold": self.config.energy_threshold,
        }


# ============================================================================
# CONVENIENCE
# ============================================================================

_default_detector: Optional[WakeWordDetector] = None


def get_wake_detector(on_wake: Optional[Callable] = None) -> WakeWordDetector:
    """Get or create the default wake word detector."""
    global _default_detector
    if _default_detector is None:
        if on_wake is None:
            raise ValueError("on_wake callback required for first initialization")
        _default_detector = WakeWordDetector(on_wake=on_wake)
    return _default_detector
