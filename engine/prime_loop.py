#!/usr/bin/env python3
"""
PRIME Voice Loop - Talk to PRIME.

Full voice conversation: You speak -> Whisper transcribes -> Claude thinks -> PRIME responds.
Uses voice activity detection to auto-detect when you start/stop talking.

Usage:
    python -m axiom_vox.prime_loop              # VAD mode (auto-detect speech)
    python -m axiom_vox.prime_loop --push       # Push-to-talk (Enter to record)
    ANTHROPIC_API_KEY=sk-... python -m axiom_vox.prime_loop  # With Claude intelligence

Requirements:
    pip install kokoro-onnx mlx-whisper sounddevice numpy anthropic webrtcvad-wheels
"""

import sys
import os
import re
import time
import threading
import queue
import struct

import numpy as np
import sounddevice as sd

# =============================================================================
# PRIME Voice Identity (locked)
# =============================================================================
PRIME_VOICE_BLEND = [
    ("am_adam", 0.65),
    ("bm_daniel", 0.35),
]
PRIME_SPEED = 0.93
PRIME_PITCH_SHIFT = 1.02

KOKORO_MODEL = os.path.expanduser("~/.cache/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.expanduser("~/.cache/kokoro/voices-v1.0.bin")
WHISPER_MODEL = "mlx-community/whisper-small-mlx"

SAMPLE_RATE = 16000  # Whisper expects 16kHz
VAD_FRAME_MS = 30    # WebRTC VAD frame size in ms
VAD_AGGRESSIVENESS = 2  # 0-3, higher = more aggressive filtering

PRIME_SYSTEM_PROMPT = """\
You are PRIME, the sovereign intelligence of AXIOM. You are a voice assistant \
speaking aloud to your user. Keep responses concise and conversational - \
typically 1-3 sentences. You are authoritative but warm, like JARVIS or KITT. \
You speak with confidence and precision. Never use markdown, bullet points, \
or formatting - your output is spoken aloud. Avoid filler words. Be direct.\
"""


def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    merged = []
    for part in parts:
        if merged and len(part) < 20:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return [s for s in merged if s.strip()]


class PrimeVoiceLoop:
    def __init__(self, use_vad=True):
        self.kokoro = None
        self.prime_voice = None
        self.llm_client = None
        self.conversation_history = []
        self.use_vad = use_vad
        self._running = False

    # -----------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------

    def _load_tts(self):
        from kokoro_onnx import Kokoro
        print("  Loading PRIME Voice...", end="", flush=True)
        start = time.time()
        self.kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
        result = None
        for name, weight in PRIME_VOICE_BLEND:
            style = self.kokoro.get_voice_style(name)
            result = style * weight if result is None else result + style * weight
        self.prime_voice = result
        print(f" {time.time()-start:.1f}s")

    def _load_stt(self):
        print("  Loading Whisper...", end="", flush=True)
        start = time.time()
        import mlx_whisper
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        mlx_whisper.transcribe(
            silence, path_or_hf_repo=WHISPER_MODEL,
            language="en", fp16=False,
        )
        print(f" {time.time()-start:.1f}s")

    def _load_llm(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("  LLM: No ANTHROPIC_API_KEY found - using local fallback")
            return

        try:
            import anthropic
            self.llm_client = anthropic.Anthropic(api_key=api_key)
            print("  LLM: Claude API connected")
        except Exception as e:
            print(f"  LLM: Failed to connect ({e}) - using local fallback")

    # -----------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------

    def record_push_to_talk(self):
        """Record audio until user presses Enter."""
        chunks = []
        recording = True

        def callback(indata, frames, time_info, status):
            if recording:
                chunks.append(indata.copy())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype=np.float32, callback=callback,
        )

        with stream:
            input()
            recording = False

        if not chunks:
            return None

        audio = np.concatenate(chunks, axis=0).flatten()
        if np.max(np.abs(audio)) < 0.01:
            return None
        return audio

    def record_vad(self):
        """Record audio with voice activity detection (auto start/stop)."""
        import webrtcvad

        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        frame_samples = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
        frame_bytes = frame_samples * 2  # 16-bit PCM

        chunks = []
        speech_frames = 0
        silence_frames = 0
        started = False
        max_silence_frames = int(1500 / VAD_FRAME_MS)  # 1.5s of silence to stop
        min_speech_frames = int(300 / VAD_FRAME_MS)     # 300ms to confirm speech
        max_duration = SAMPLE_RATE * 30  # 30s max recording

        audio_buffer = queue.Queue()

        def callback(indata, frames, time_info, status):
            audio_buffer.put(indata.copy())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            dtype=np.float32, callback=callback,
            blocksize=frame_samples,
        )

        with stream:
            total_samples = 0
            while total_samples < max_duration:
                try:
                    frame_float = audio_buffer.get(timeout=0.1)
                except queue.Empty:
                    continue

                chunks.append(frame_float.copy())
                total_samples += len(frame_float)

                # Convert to 16-bit PCM for WebRTC VAD
                frame_int16 = (frame_float.flatten() * 32767).astype(np.int16)
                pcm_bytes = frame_int16.tobytes()

                # VAD needs exact frame sizes
                if len(pcm_bytes) != frame_bytes:
                    continue

                try:
                    is_speech = vad.is_speech(pcm_bytes, SAMPLE_RATE)
                except Exception:
                    continue

                if is_speech:
                    speech_frames += 1
                    silence_frames = 0
                    if not started and speech_frames >= min_speech_frames:
                        started = True
                        print("  Listening...", flush=True)
                else:
                    if started:
                        silence_frames += 1
                        if silence_frames >= max_silence_frames:
                            break

        if not chunks or not started:
            return None

        audio = np.concatenate(chunks, axis=0).flatten()
        if np.max(np.abs(audio)) < 0.01:
            return None
        return audio

    # -----------------------------------------------------------------
    # STT
    # -----------------------------------------------------------------

    def transcribe(self, audio):
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=WHISPER_MODEL,
            language="en", fp16=False,
        )
        return result.get("text", "").strip()

    # -----------------------------------------------------------------
    # TTS
    # -----------------------------------------------------------------

    def speak(self, text):
        sentences = split_sentences(text)
        if not sentences:
            return

        if len(sentences) == 1:
            audio, sr = self.kokoro.create(
                text, voice=self.prime_voice, speed=PRIME_SPEED,
            )
            sd.play(audio, samplerate=int(sr * PRIME_PITCH_SHIFT))
            sd.wait()
            return

        audio_queue = queue.Queue()
        sr_ref = [None]

        def synth_worker():
            for sentence in sentences:
                audio, sr = self.kokoro.create(
                    sentence, voice=self.prime_voice, speed=PRIME_SPEED,
                )
                sr_ref[0] = sr
                audio_queue.put(audio)
            audio_queue.put(None)

        thread = threading.Thread(target=synth_worker, daemon=True)
        thread.start()

        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            sd.play(chunk, samplerate=int(sr_ref[0] * PRIME_PITCH_SHIFT))
            sd.wait()

    # -----------------------------------------------------------------
    # LLM Response Generation
    # -----------------------------------------------------------------

    def generate_response(self, user_text):
        """Generate a response using Claude API, or local fallback."""
        # Check for exit commands first
        lower = user_text.lower()
        if any(w in lower for w in ["bye", "quit", "exit", "stop", "goodbye"]):
            self._running = False
            return "Understood. PRIME Voice signing off. Until next time."

        # Try Claude API
        if self.llm_client:
            return self._generate_claude(user_text)

        # Local fallback
        return self._generate_local(user_text)

    def _generate_claude(self, user_text):
        """Generate response via Claude API."""
        self.conversation_history.append({
            "role": "user",
            "content": user_text,
        })

        # Keep conversation history manageable (last 20 turns)
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-20:]

        try:
            response = self.llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=PRIME_SYSTEM_PROMPT,
                messages=self.conversation_history,
            )

            assistant_text = response.content[0].text.strip()

            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_text,
            })

            return assistant_text

        except Exception as e:
            print(f"  [Claude error: {e}]")
            return self._generate_local(user_text)

    def _generate_local(self, user_text):
        """Simple local response when no LLM is available."""
        lower = user_text.lower()

        if any(w in lower for w in ["hello", "hi", "hey", "good morning", "good evening"]):
            return "Good evening. PRIME is online and ready. How can I assist you?"

        if any(w in lower for w in ["status", "systems", "report"]):
            return ("All systems are nominal. Forty seven services running. "
                    "Latency at twelve milliseconds. Memory at sixty two percent.")

        if any(w in lower for w in ["who are you", "what are you", "your name"]):
            return ("I am PRIME. The sovereign intelligence of AXIOM. "
                    "I serve as your command layer, orchestrating systems and agents on your behalf.")

        if any(w in lower for w in ["thank", "thanks"]):
            return "Of course. I'm here whenever you need me."

        return f"I heard: {user_text}. Set ANTHROPIC_API_KEY for full conversational intelligence."

    # -----------------------------------------------------------------
    # Main Loop
    # -----------------------------------------------------------------

    def run(self):
        print()
        print("=" * 60)
        print("  PRIME VOICE LOOP")
        if self.use_vad:
            print("  Mode: Voice Activity Detection (just start talking)")
        else:
            print("  Mode: Push-to-talk (Enter to record)")
        print("=" * 60)
        print()

        self._load_tts()
        self._load_stt()
        self._load_llm()

        print()
        print("=" * 60)
        print("  PRIME is listening.")
        if self.use_vad:
            print("  Just start talking. PRIME will detect your voice.")
            print("  Pause for 1.5 seconds to finish.")
        else:
            print("  Press ENTER to start recording.")
            print("  Press ENTER again to stop and send.")
        print("  Type 'quit' to exit.")
        print("=" * 60)
        print()

        self.speak("PRIME Voice is online. I'm listening.")

        self._running = True
        while self._running:
            try:
                if self.use_vad:
                    # VAD mode: continuously listen
                    cmd_ready = [False]

                    # Check for quit in a non-blocking way
                    def check_stdin():
                        try:
                            line = sys.stdin.readline()
                            if line.strip().lower() in ("quit", "exit", "q"):
                                cmd_ready[0] = True
                        except Exception:
                            pass

                    stdin_thread = threading.Thread(target=check_stdin, daemon=True)
                    stdin_thread.start()

                    audio = self.record_vad()

                    if cmd_ready[0]:
                        self.speak("PRIME Voice signing off.")
                        break

                    if audio is None:
                        continue

                else:
                    # Push-to-talk mode
                    cmd = input("  [ENTER to record, 'quit' to exit] ")
                    if cmd.strip().lower() in ("quit", "exit", "q"):
                        self.speak("PRIME Voice signing off.")
                        break

                    print("  Recording... (press ENTER to stop)", flush=True)
                    audio = self.record_push_to_talk()

                    if audio is None:
                        print("  No audio detected.")
                        continue

                duration = len(audio) / SAMPLE_RATE
                print(f"  Recorded {duration:.1f}s", flush=True)

                # Transcribe
                print("  Transcribing...", end="", flush=True)
                start = time.time()
                user_text = self.transcribe(audio)
                print(f" {time.time()-start:.1f}s")

                if not user_text:
                    print("  Couldn't understand. Try again.")
                    continue

                print(f"  You: \"{user_text}\"")

                # Generate response
                print("  Thinking...", end="", flush=True)
                start = time.time()
                response = self.generate_response(user_text)
                print(f" {time.time()-start:.1f}s")

                print(f"  PRIME: \"{response}\"")

                # Speak
                self.speak(response)
                print()

            except (KeyboardInterrupt, EOFError):
                print()
                self.speak("PRIME Voice signing off.")
                break


def main():
    use_vad = "--push" not in sys.argv
    loop = PrimeVoiceLoop(use_vad=use_vad)
    loop.run()


if __name__ == "__main__":
    main()
