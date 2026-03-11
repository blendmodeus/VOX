#!/usr/bin/env python3
"""
PRIME Say - The voice of AXIOM.

Speaks text in PRIME's locked voice identity using Kokoro TTS.
Voice: am_adam:0.65 + bm_daniel:0.35 @ 0.93x speed.
Streams sentence-by-sentence for low perceived latency.

Usage:
    python -m axiom_vox.prime_say "Hello, this is PRIME speaking."
    echo "Some text" | python -m axiom_vox.prime_say
"""

import sys
import os
import re
import time
import threading
import queue

# =============================================================================
# PRIME Voice Identity (locked)
# =============================================================================
PRIME_VOICE_BLEND = [
    ("am_adam", 0.65),     # Warm, clean American base
    ("bm_daniel", 0.35),   # British crispness, precision
]
PRIME_SPEED = 0.93  # Slightly measured for gravitas
PRIME_PITCH_SHIFT = 1.02  # 2% pitch up (resample ratio)

MODEL_PATH = os.path.expanduser("~/.cache/kokoro/kokoro-v1.0.onnx")
VOICES_PATH = os.path.expanduser("~/.cache/kokoro/voices-v1.0.bin")


def split_sentences(text):
    """Split text into sentences for progressive synthesis."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    merged = []
    for part in parts:
        if merged and len(part) < 20:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)
    return [s for s in merged if s.strip()]


def build_prime_voice(kokoro):
    """Build PRIME's blended voice vector."""
    result = None
    for name, weight in PRIME_VOICE_BLEND:
        style = kokoro.get_voice_style(name)
        result = style * weight if result is None else result + style * weight
    return result


def main():
    # Get text from args or stdin
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:]).strip()
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        print("Usage: python -m axiom_vox.prime_say \"text to speak\"")
        sys.exit(1)

    if not text:
        sys.exit(0)

    import numpy as np
    import sounddevice as sd
    from kokoro_onnx import Kokoro

    # Load Kokoro (sub-1s load time)
    print(f"  Loading PRIME Voice...", end="", flush=True)
    load_start = time.time()
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
    prime_voice = build_prime_voice(kokoro)
    print(f" {time.time() - load_start:.1f}s")

    sentences = split_sentences(text)

    if len(sentences) <= 1:
        # Short text - synthesize and play directly
        print(f"  \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

        synth_start = time.time()
        audio, sr = kokoro.create(text, voice=prime_voice, speed=PRIME_SPEED)
        synth_time = time.time() - synth_start
        duration = len(audio) / sr

        print(f"  {duration:.1f}s audio in {synth_time:.1f}s "
              f"(RTF: {synth_time/duration:.2f}x)")
        sd.play(audio, samplerate=int(sr * PRIME_PITCH_SHIFT))
        sd.wait()
        return

    # Multi-sentence streaming: synthesize and play concurrently
    print(f"  Streaming {len(sentences)} sentences...")
    print(f"  \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    print()

    audio_queue = queue.Queue()
    total_synth_time = 0.0
    total_audio_duration = 0.0
    sample_rate_ref = [None]

    def synth_worker():
        nonlocal total_synth_time, total_audio_duration
        for i, sentence in enumerate(sentences):
            s_start = time.time()
            audio, sr = kokoro.create(sentence, voice=prime_voice, speed=PRIME_SPEED)
            s_time = time.time() - s_start

            duration = len(audio) / sr
            total_synth_time += s_time
            total_audio_duration += duration
            sample_rate_ref[0] = sr

            print(f"    [{i+1}/{len(sentences)}] "
                  f"{duration:.1f}s in {s_time:.1f}s "
                  f"\"{sentence[:50]}{'...' if len(sentence) > 50 else ''}\"")

            audio_queue.put(audio)

        audio_queue.put(None)  # sentinel

    # Start synthesis in background thread
    wall_start = time.time()
    synth_thread = threading.Thread(target=synth_worker, daemon=True)
    synth_thread.start()

    # Play audio chunks as they arrive
    first_chunk = True
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break

        if first_chunk:
            first_latency = time.time() - wall_start
            print(f"\n  First audio in {first_latency:.1f}s")
            first_chunk = False

        sd.play(audio_chunk, samplerate=int(sample_rate_ref[0] * PRIME_PITCH_SHIFT))
        sd.wait()

    wall_time = time.time() - wall_start

    print(f"\n  Total: {total_audio_duration:.1f}s audio, "
          f"{wall_time:.1f}s wall time "
          f"(RTF: {wall_time/total_audio_duration:.2f}x)")


if __name__ == "__main__":
    main()
