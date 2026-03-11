#!/usr/bin/env python3
"""
PRIME Say (Chatterbox) — Voice-cloned PRIME speech via Chatterbox Turbo.

Speaks text using a cloned voice from a reference audio clip.
This is the high-quality tier — use prime_say.py for the fast Kokoro tier.

Usage:
    python -m axiom_vox.prime_say_cb "Hello, this is PRIME speaking."
    python -m axiom_vox.prime_say_cb --ref path/to/voice.wav "Custom voice"
    echo "Some text" | python -m axiom_vox.prime_say_cb

Setup (first time):
    python -m axiom_vox.prime_say_cb --set-ref path/to/prime_voice.wav
"""

import sys
import os
import re
import time
import argparse
import threading
import queue


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


def main():
    parser = argparse.ArgumentParser(
        description="PRIME Voice (Chatterbox Turbo) — Voice-cloned TTS"
    )
    parser.add_argument("text", nargs="*", help="Text to speak")
    parser.add_argument(
        "--ref", type=str, default=None,
        help="Path to reference audio WAV (5-15s) for voice cloning"
    )
    parser.add_argument(
        "--set-ref", type=str, default=None,
        help="Set a new default PRIME reference voice"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda, mps, cpu (auto-detected)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Save output to WAV file instead of playing"
    )
    parser.add_argument(
        "--exaggeration", type=float, default=0.5,
        help="Emotion exaggeration (0.0-1.0, default 0.5)"
    )
    parser.add_argument(
        "--vectorize", action="store_true",
        help="Distill the reference voice into a permanent .pt vector for speed"
    )
    args = parser.parse_args()

    # Handle --set-ref: just store the reference and exit
    if args.set_ref:
        from axiom_vox.chatterbox_engine import ChatterboxEngine
        dest = ChatterboxEngine.setup_prime_reference(args.set_ref)
        print(f"  ✓ PRIME reference voice set: {dest}")
        sys.exit(0)

    # Get text from args or stdin
    if args.text:
        text = " ".join(args.text).strip()
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        parser.print_help()
        sys.exit(1)

    if not text and not args.vectorize:
        sys.exit(0)

    import numpy as np
    import sounddevice as sd
    from axiom_vox.chatterbox_engine import ChatterboxEngine

    # Initialize engine
    print(f"  Loading Chatterbox Turbo...", end="", flush=True)
    load_start = time.time()
    engine = ChatterboxEngine(device=args.device, ref_audio=args.ref)

    if not engine.load():
        print(" FAILED")
        print("  Install: pip install chatterbox-tts")
        sys.exit(1)

    print(f" {time.time() - load_start:.1f}s")

    # Handle --vectorize
    if args.vectorize:
        ref = args.ref or engine.ref_audio
        if not os.path.exists(ref):
            print(f"  ✗ Cannot vectorize: reference audio not found at {ref}")
            sys.exit(1)
        
        from axiom_vox.chatterbox_engine import PRIME_VECTOR_DEFAULT
        print(f"  Distilling {os.path.basename(ref)}...", end="", flush=True)
        v_start = time.time()
        engine.vectorize(ref, PRIME_VECTOR_DEFAULT)
        print(f" DONE ({time.time() - v_start:.1f}s)")
        print(f"  ✓ Concentrated voice vector saved to: {PRIME_VECTOR_DEFAULT}")
        if not text:
            sys.exit(0)

    ref_label = args.ref or engine.ref_audio
    if os.path.exists(ref_label):
        print(f"  Reference: {os.path.basename(ref_label)}")
    else:
        print(f"  ⚠ No reference audio — generating without voice cloning")

    sentences = split_sentences(text)

    if len(sentences) <= 1 or args.out:
        # Single shot
        print(f'  "{text[:80]}{"..." if len(text) > 80 else ""}"')

        synth_start = time.time()
        wav_bytes = engine.generate(
            text,
            ref_audio=args.ref,
            out=args.out,
            exaggeration=args.exaggeration,
        )
        synth_time = time.time() - synth_start

        if wav_bytes is None:
            print("  ✗ Synthesis failed")
            sys.exit(1)

        if args.out:
            print(f"  ✓ Saved to {args.out} ({synth_time:.1f}s)")
        else:
            import soundfile as sf
            import io
            audio, sr = sf.read(io.BytesIO(wav_bytes))
            duration = len(audio) / sr
            print(f"  {duration:.1f}s audio in {synth_time:.1f}s "
                  f"(RTF: {synth_time/duration:.2f}x)")
            sd.play(audio, samplerate=sr)
            sd.wait()
        return

    # Multi-sentence streaming
    print(f"  Streaming {len(sentences)} sentences...")
    print(f'  "{text[:80]}{"..." if len(text) > 80 else ""}"')
    print()

    audio_queue = queue.Queue()
    total_synth_time = 0.0
    total_audio_duration = 0.0
    sample_rate_ref = [None]

    def synth_worker():
        nonlocal total_synth_time, total_audio_duration
        import soundfile as sf
        import io

        for i, sentence in enumerate(sentences):
            s_start = time.time()
            wav_bytes = engine.generate(
                sentence,
                ref_audio=args.ref,
                exaggeration=args.exaggeration,
            )
            s_time = time.time() - s_start

            if wav_bytes is None:
                print(f"    [{i+1}] FAILED: {sentence[:40]}")
                continue

            audio, sr = sf.read(io.BytesIO(wav_bytes))
            duration = len(audio) / sr
            total_synth_time += s_time
            total_audio_duration += duration
            sample_rate_ref[0] = sr

            print(f"    [{i+1}/{len(sentences)}] "
                  f"{duration:.1f}s in {s_time:.1f}s "
                  f'"{sentence[:50]}{"..." if len(sentence) > 50 else ""}"')

            audio_queue.put((audio, sr))

        audio_queue.put(None)

    wall_start = time.time()
    synth_thread = threading.Thread(target=synth_worker, daemon=True)
    synth_thread.start()

    first_chunk = True
    while True:
        item = audio_queue.get()
        if item is None:
            break

        audio, sr = item

        if first_chunk:
            first_latency = time.time() - wall_start
            print(f"\n  First audio in {first_latency:.1f}s")
            first_chunk = False

        sd.play(audio, samplerate=sr)
        sd.wait()

    wall_time = time.time() - wall_start

    if total_audio_duration > 0:
        print(f"\n  Total: {total_audio_duration:.1f}s audio, "
              f"{wall_time:.1f}s wall time "
              f"(RTF: {wall_time/total_audio_duration:.2f}x)")
    else:
        print("\n  No audio generated.")


if __name__ == "__main__":
    main()
